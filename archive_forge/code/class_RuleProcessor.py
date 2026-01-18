import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
class RuleProcessor(object):
    """A class to process assertions and mapping rules."""

    class _EvalType(object):
        """Mapping rule evaluation types."""
        ANY_ONE_OF = 'any_one_of'
        NOT_ANY_OF = 'not_any_of'
        BLACKLIST = 'blacklist'
        WHITELIST = 'whitelist'

    def __init__(self, mapping_id, rules):
        """Initialize RuleProcessor.

        Example rules can be found at:
        :class:`keystone.tests.mapping_fixtures`

        :param mapping_id: id for the mapping
        :type mapping_id: string
        :param rules: rules from a mapping
        :type rules: dict

        """
        self.mapping_id = mapping_id
        self.rules = rules

    def process(self, assertion_data):
        """Transform assertion to a dictionary.

        The dictionary contains mapping of user name and group ids
        based on mapping rules.

        This function will iterate through the mapping rules to find
        assertions that are valid.

        :param assertion_data: an assertion containing values from an IdP
        :type assertion_data: dict

        Example assertion_data::

            {
                'Email': 'testacct@example.com',
                'UserName': 'testacct',
                'FirstName': 'Test',
                'LastName': 'Account',
                'orgPersonType': 'Tester'
            }

        :returns: dictionary with user and group_ids

        The expected return structure is::

            {
                'name': 'foobar',
                'group_ids': ['abc123', 'def456'],
                'group_names': [
                    {
                        'name': 'group_name_1',
                        'domain': {
                            'name': 'domain1'
                        }
                    },
                    {
                        'name': 'group_name_1_1',
                        'domain': {
                            'name': 'domain1'
                        }
                    },
                    {
                        'name': 'group_name_2',
                        'domain': {
                            'id': 'xyz132'
                        }
                    }
                ]
            }

        """
        LOG.debug('assertion data: %s', assertion_data)
        assertion = {n: v.split(';') for n, v in assertion_data.items() if isinstance(v, str)}
        LOG.debug('assertion: %s', assertion)
        identity_values = []
        LOG.debug('rules: %s', self.rules)
        for rule in self.rules:
            direct_maps = self._verify_all_requirements(rule['remote'], assertion)
            if direct_maps is None:
                continue
            if not direct_maps:
                identity_values += rule['local']
            else:
                for local in rule['local']:
                    new_local = self._update_local_mapping(local, direct_maps)
                    identity_values.append(new_local)
        LOG.debug('identity_values: %s', identity_values)
        mapped_properties = self._transform(identity_values)
        LOG.debug('mapped_properties: %s', mapped_properties)
        return mapped_properties

    @staticmethod
    def _ast_literal_eval(value):
        try:
            values = ast.literal_eval(value)
            if not isinstance(values, list):
                raise ValueError
        except (ValueError, SyntaxError):
            values = [value]
        return values

    def _normalize_groups(self, identity_value):
        if 'name' in identity_value['groups']:
            group_names_list = self._ast_literal_eval(identity_value['groups'])

            def convert_json(group):
                if group.startswith('JSON:'):
                    return jsonutils.loads(group.lstrip('JSON:'))
                return group
            group_dicts = [convert_json(g) for g in group_names_list]
            for g in group_dicts:
                if 'domain' not in g:
                    msg = _("Invalid rule: %(identity_value)s. Both 'groups' and 'domain' keywords must be specified.")
                    msg = msg % {'identity_value': identity_value}
                    raise exception.ValidationError(msg)
        else:
            if 'domain' not in identity_value:
                msg = _("Invalid rule: %(identity_value)s. Both 'groups' and 'domain' keywords must be specified.")
                msg = msg % {'identity_value': identity_value}
                raise exception.ValidationError(msg)
            group_names_list = self._ast_literal_eval(identity_value['groups'])
            domain = identity_value['domain']
            group_dicts = [{'name': name, 'domain': domain} for name in group_names_list]
        return group_dicts

    def normalize_user(self, user, default_mapping_domain):
        """Parse and validate user mapping."""
        if user.get('type') is None:
            user['type'] = UserType.EPHEMERAL
        if user.get('type') not in (UserType.EPHEMERAL, UserType.LOCAL):
            msg = _('User type %s not supported') % user.get('type')
            raise exception.ValidationError(msg)

    def extract_groups(self, groups_by_domain):
        for groups in list(groups_by_domain.values()):
            for group in list({g['name']: g for g in groups}.values()):
                yield group

    def _transform(self, identity_values):
        """Transform local mappings, to an easier to understand format.

        Transform the incoming array to generate the return value for
        the process function. Generating content for Keystone tokens will
        be easier if some pre-processing is done at this level.

        :param identity_values: local mapping from valid evaluations
        :type identity_values: array of dict

        Example identity_values::

            [
                {
                    'group': {'id': '0cd5e9'},
                    'user': {
                        'email': 'bob@example.com'
                    },
                },
                {
                    'groups': ['member', 'admin', tester'],
                    'domain': {
                        'name': 'default_domain'
                    }
                },
                {
                    'group_ids': ['abc123', 'def456', '0cd5e9']
                }
            ]

        :returns: dictionary with user name, group_ids and group_names.
        :rtype: dict

        """
        user = {}
        group_ids = set()
        group_names = list()
        groups_by_domain = dict()
        projects = []
        if not identity_values:
            msg = 'Could not map any federated user properties to identity values. Check debug logs or the mapping used for additional details.'
            tr_msg = _('Could not map any federated user properties to identity values. Check debug logs or the mapping used for additional details.')
            LOG.warning(msg)
            raise exception.ValidationError(tr_msg)
        for identity_value in identity_values:
            if 'user' in identity_value:
                if user:
                    LOG.warning('Ignoring user [%s]', identity_value.get('user'))
                else:
                    user = identity_value.get('user')
            if 'group' in identity_value:
                group = identity_value['group']
                if 'id' in group:
                    group_ids.add(group['id'])
                elif 'name' in group:
                    groups = self.process_group_by_name(group, groups_by_domain)
                    group_names.extend(groups)
            if 'groups' in identity_value:
                group_dicts = self._normalize_groups(identity_value)
                group_names.extend(group_dicts)
            if 'group_ids' in identity_value:
                group_ids.update(self._ast_literal_eval(identity_value['group_ids']))
            if 'projects' in identity_value:
                projects = self.extract_projects(identity_value)
        self.normalize_user(user, identity_value.get('domain'))
        return {'user': user, 'group_ids': list(group_ids), 'group_names': group_names, 'projects': projects}

    def process_group_by_name(self, group, groups_by_domain):
        domain = group['domain'].get('name') or group['domain'].get('id')
        groups_by_domain.setdefault(domain, list()).append(group)
        return self.extract_groups(groups_by_domain)

    def extract_projects(self, identity_value):
        return identity_value.get('projects', [])

    def _update_local_mapping(self, local, direct_maps):
        """Replace any {0}, {1} ... values with data from the assertion.

        :param local: local mapping reference that needs to be updated
        :type local: dict
        :param direct_maps: identity values used to update local
        :type direct_maps: keystone.federation.utils.DirectMaps

        Example local::

            {'user': {'name': '{0} {1}', 'email': '{2}'}}

        Example direct_maps::

            [['Bob'], ['Thompson'], ['bob@example.com']]

        :returns: new local mapping reference with replaced values.

        The expected return structure is::

            {'user': {'name': 'Bob Thompson', 'email': 'bob@example.org'}}

        :raises keystone.exception.DirectMappingError: when referring to a
            remote match from a local section of a rule

        """
        LOG.debug('direct_maps: %s', direct_maps)
        LOG.debug('local: %s', local)
        new = {}
        for k, v in local.items():
            if isinstance(v, dict):
                new_value = self._update_local_mapping(v, direct_maps)
            elif isinstance(v, list):
                new_value = [self._update_local_mapping(item, direct_maps) for item in v]
            else:
                try:
                    new_value = v.format(*direct_maps)
                except IndexError:
                    raise exception.DirectMappingError(mapping_id=self.mapping_id)
            new[k] = new_value
        return new

    def _verify_all_requirements(self, requirements, assertion):
        """Compare remote requirements of a rule against the assertion.

        If a value of ``None`` is returned, the rule with this assertion
        doesn't apply.
        If an array of zero length is returned, then there are no direct
        mappings to be performed, but the rule is valid.
        Otherwise, then it will first attempt to filter the values according
        to blacklist or whitelist rules and finally return the values in
        order, to be directly mapped.

        :param requirements: list of remote requirements from rules
        :type requirements: list

        Example requirements::

            [
                {
                    "type": "UserName"
                },
                {
                    "type": "orgPersonType",
                    "any_one_of": [
                        "Customer"
                    ]
                },
                {
                    "type": "ADFS_GROUPS",
                    "whitelist": [
                        "g1", "g2", "g3", "g4"
                    ]
                }
            ]

        :param assertion: dict of attributes from an IdP
        :type assertion: dict

        Example assertion::

            {
                'UserName': ['testacct'],
                'LastName': ['Account'],
                'orgPersonType': ['Tester'],
                'Email': ['testacct@example.com'],
                'FirstName': ['Test'],
                'ADFS_GROUPS': ['g1', 'g2']
            }

        :returns: identity values used to update local
        :rtype: keystone.federation.utils.DirectMaps or None

        """
        direct_maps = DirectMaps()
        for requirement in requirements:
            requirement_type = requirement['type']
            direct_map_values = assertion.get(requirement_type)
            regex = requirement.get('regex', False)
            if not direct_map_values:
                return None
            any_one_values = requirement.get(self._EvalType.ANY_ONE_OF)
            if any_one_values is not None:
                if self._evaluate_requirement(any_one_values, direct_map_values, self._EvalType.ANY_ONE_OF, regex):
                    continue
                else:
                    return None
            not_any_values = requirement.get(self._EvalType.NOT_ANY_OF)
            if not_any_values is not None:
                if self._evaluate_requirement(not_any_values, direct_map_values, self._EvalType.NOT_ANY_OF, regex):
                    continue
                else:
                    return None
            blacklisted_values = requirement.get(self._EvalType.BLACKLIST)
            whitelisted_values = requirement.get(self._EvalType.WHITELIST)
            if blacklisted_values is not None:
                direct_map_values = self._evaluate_requirement(blacklisted_values, direct_map_values, self._EvalType.BLACKLIST, regex)
            elif whitelisted_values is not None:
                direct_map_values = self._evaluate_requirement(whitelisted_values, direct_map_values, self._EvalType.WHITELIST, regex)
            direct_maps.add(direct_map_values)
            LOG.debug('updating a direct mapping: %s', direct_map_values)
        return direct_maps

    def _evaluate_values_by_regex(self, values, assertion_values):
        return [assertion for assertion in assertion_values if any([re.search(regex, assertion) for regex in values])]

    def _evaluate_requirement(self, values, assertion_values, eval_type, regex):
        """Evaluate the incoming requirement and assertion.

        Filter the incoming assertions against the requirement values. If regex
        is specified, the assertion list is filtered by checking if any of the
        requirement regexes matches. Otherwise, the list is filtered by string
        equality with any of the allowed values.

        Once the assertion values are filtered, the output is determined by the
        evaluation type:
            any_one_of: return True if there are any matches, False otherwise
            not_any_of: return True if there are no matches, False otherwise
            blacklist: return the incoming values minus any matches
            whitelist: return only the matched values

        :param values: list of allowed values, defined in the requirement
        :type values: list
        :param assertion_values: The values from the assertion to evaluate
        :type assertion_values: list/string
        :param eval_type: determine how to evaluate requirements
        :type eval_type: string
        :param regex: perform evaluation with regex
        :type regex: boolean

        :returns: list of filtered assertion values (if evaluation type is
                  'blacklist' or 'whitelist'), or boolean indicating if the
                  assertion values fulfill the requirement (if evaluation type
                  is 'any_one_of' or 'not_any_of')

        """
        if regex:
            matches = self._evaluate_values_by_regex(values, assertion_values)
        else:
            matches = set(values).intersection(set(assertion_values))
        if eval_type == self._EvalType.ANY_ONE_OF:
            return bool(matches)
        elif eval_type == self._EvalType.NOT_ANY_OF:
            return not bool(matches)
        elif eval_type == self._EvalType.BLACKLIST:
            return list(set(assertion_values).difference(set(matches)))
        elif eval_type == self._EvalType.WHITELIST:
            return list(matches)
        else:
            raise exception.UnexpectedError(_('Unexpected evaluation type "%(eval_type)s"') % {'eval_type': eval_type})