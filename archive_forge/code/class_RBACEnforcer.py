import functools
import flask
from oslo_log import log
from oslo_policy import opts
from oslo_policy import policy as common_policy
from oslo_utils import strutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import policies
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
class RBACEnforcer(object):
    """Enforce RBAC on API calls."""
    __shared_state__ = {}
    __ENFORCER = None
    ACTION_STORE_ATTR = 'keystone:RBAC:action_name'
    suppress_deprecation_warnings = False

    def __init__(self):
        self.__dict__ = self.__shared_state__

    def _check_deprecated_rule(self, action):

        def _name_is_changing(rule):
            deprecated_rule = rule.deprecated_rule
            return deprecated_rule and deprecated_rule.name != rule.name and (deprecated_rule.name in self._enforcer.file_rules)

        def _check_str_is_changing(rule):
            deprecated_rule = rule.deprecated_rule
            return deprecated_rule and deprecated_rule.check_str != rule.check_str and (rule.name not in self._enforcer.file_rules)

        def _is_deprecated_for_removal(rule):
            return rule.deprecated_for_removal and rule.name in self._enforcer.file_rules

        def _emit_warning():
            if not self._enforcer._warning_emitted:
                LOG.warning('Deprecated policy rules found. Use oslopolicy-policy-generator and oslopolicy-policy-upgrade to detect and resolve deprecated policies in your configuration.')
                self._enforcer._warning_emitted = True
        registered_rule = self._enforcer.registered_rules.get(action)
        if not registered_rule:
            return
        if _name_is_changing(registered_rule) or _check_str_is_changing(registered_rule) or _is_deprecated_for_removal(registered_rule):
            _emit_warning()

    def _enforce(self, credentials, action, target, do_raise=True):
        """Verify that the action is valid on the target in this context.

        This method is for cases that exceed the base enforcer
        functionality (notably for compatibility with `@protected` style
        decorators.

        :param credentials: user credentials
        :param action: string representing the action to be checked, which
                       should be colon separated for clarity.
        :param target: dictionary representing the object of the action for
                       object creation this should be a dictionary
                       representing the location of the object e.g.
                       {'project_id': object.project_id}
        :raises keystone.exception.Forbidden: If verification fails.

        Actions should be colon separated for clarity. For example:

        * identity:list_users
        """
        extra = {}
        if do_raise:
            extra.update(exc=exception.ForbiddenAction, action=action, do_raise=do_raise)
        try:
            result = self._enforcer.enforce(rule=action, target=target, creds=credentials, **extra)
            self._check_deprecated_rule(action)
            return result
        except common_policy.InvalidScope:
            raise exception.ForbiddenAction(action=action)

    def _reset(self):
        self.__ENFORCER = None

    @property
    def _enforcer(self):
        if self.__ENFORCER is None:
            self.__ENFORCER = common_policy.Enforcer(CONF)
            if flask.has_request_context():
                self.__ENFORCER.suppress_deprecation_warnings = True
            if self.suppress_deprecation_warnings:
                self.__ENFORCER.suppress_deprecation_warnings = True
            self.register_rules(self.__ENFORCER)
            self.__ENFORCER._warning_emitted = False
        return self.__ENFORCER

    @staticmethod
    def _extract_filter_values(filters):
        """Extract filter data from query params for RBAC enforcement."""
        filters = filters or []
        target = {i: flask.request.args[i] for i in filters if i in flask.request.args}
        if target:
            if LOG.logger.getEffectiveLevel() <= log.DEBUG:
                LOG.debug('RBAC: Adding query filter params (%s)', ', '.join(['%s=%s' % (k, v) for k, v in target.items()]))
        return target

    @staticmethod
    def _extract_member_target_data(member_target_type, member_target):
        """Build some useful target data.

        :param member_target_type: what type of target, e.g. 'user'
        :type member_target_type: str or None
        :param member_target: reference of the target data
        :type member_target: dict or None
        :returns: constructed target dict or empty dict
        :rtype: dict
        """
        ret_dict = {}
        if member_target is not None and member_target_type is None or (member_target is None and member_target_type is not None):
            LOG.warning('RBAC: Unknown target type or target reference. Rejecting as unauthorized. (member_target_type=%(target_type)r, member_target=%(target_ref)r)', {'target_type': member_target_type, 'target_ref': member_target})
            return ret_dict
        if member_target is not None and member_target_type is not None:
            ret_dict['target'] = {member_target_type: member_target}
        elif flask.request.endpoint:
            resource = flask.current_app.view_functions[flask.request.endpoint].view_class
            try:
                member_name = getattr(resource, 'member_key', None)
            except ValueError:
                member_name = None
            func = getattr(resource, 'get_member_from_driver', None)
            if member_name is not None and callable(func):
                key = '%s_id' % member_name
                if key in (flask.request.view_args or {}):
                    ret_dict['target'] = {member_name: func(flask.request.view_args[key])}
        return ret_dict

    @staticmethod
    def _extract_policy_check_credentials():
        return flask.request.environ.get(authorization.AUTH_CONTEXT_ENV, {})

    @classmethod
    def _extract_subject_token_target_data(cls):
        ret_dict = {}
        window_seconds = 0
        target = 'token'
        subject_token = flask.request.headers.get('X-Subject-Token')
        access_rules_support = flask.request.headers.get(authorization.ACCESS_RULES_HEADER)
        if subject_token is not None:
            allow_expired = strutils.bool_from_string(flask.request.args.get('allow_expired', False), default=False)
            if allow_expired:
                window_seconds = CONF.token.allow_expired_window
            token = PROVIDER_APIS.token_provider_api.validate_token(subject_token, window_seconds=window_seconds, access_rules_support=access_rules_support)
            ret_dict[target] = {}
            ret_dict[target]['user_id'] = token.user_id
            try:
                user_domain_id = token.user['domain_id']
            except exception.UnexpectedError:
                user_domain_id = None
            if user_domain_id:
                ret_dict[target].setdefault('user', {})
                ret_dict[target]['user'].setdefault('domain', {})
                ret_dict[target]['user']['domain']['id'] = user_domain_id
        return ret_dict

    @staticmethod
    def _get_oslo_req_context():
        return flask.request.environ.get(context.REQUEST_CONTEXT_ENV, None)

    @classmethod
    def _assert_is_authenticated(cls):
        ctx = cls._get_oslo_req_context()
        if ctx is None:
            LOG.warning('RBAC: Error reading the request context generated by the Auth Middleware (there is no context). Rejecting request as unauthorized.')
            raise exception.Unauthorized(_('Internal error processing authentication and authorization.'))
        if not ctx.authenticated:
            raise exception.Unauthorized(_('auth_context did not decode anything useful'))

    @classmethod
    def _shared_admin_auth_token_set(cls):
        ctx = cls._get_oslo_req_context()
        return getattr(ctx, 'is_admin', False)

    @classmethod
    def enforce_call(cls, enforcer=None, action=None, target_attr=None, member_target_type=None, member_target=None, filters=None, build_target=None):
        """Enforce RBAC on the current request.

        This will do some legwork and then instantiate the Enforcer if an
        enforcer is not passed in.

        :param enforcer: A pre-instantiated Enforcer object (optional)
        :type enforcer: :class:`RBACEnforcer`
        :param action: the name of the rule/policy enforcement to be checked
                       against, e.g. `identity:get_user` (optional may be
                       replaced by decorating the method/function with
                       `policy_enforcer_action`.
        :type action: str
        :param target_attr: complete override of the target data. This will
                            replace all other generated target data meaning
                            `member_target_type` and `member_target` are
                            ignored. This will also prevent extraction of
                            data from the X-Subject-Token. The `target` dict
                            should contain a series of key-value pairs such
                            as `{'user': user_ref_dict}`.
        :type target_attr: dict
        :param member_target_type: the type of the target, e.g. 'user'. Both
                                   this and `member_target` must be passed if
                                   either is passed.
        :type member_target_type: str
        :param member_target: the (dict form) reference of the member object.
                              Both this and `member_target_type` must be passed
                              if either is passed.
        :type member_target: dict
        :param filters: A variable number of optional string filters, these are
                        used to extract values from the query params. The
                        filters are added to the request data that is passed to
                        the enforcer and may be used to determine policy
                        action. In practice these are mainly supplied in the
                        various "list" APIs and are un-used in the default
                        supplied policies.
        :type filters: iterable
        :param build_target: A function to build the target for enforcement.
                             This is explicitly done after authentication
                             in order to not leak existance data before
                             auth.
        :type build_target: function
        """
        policy_dict = {}
        action = action or getattr(flask.g, cls.ACTION_STORE_ATTR, None)
        if action not in _POSSIBLE_TARGET_ACTIONS:
            LOG.warning('RBAC: Unknown enforcement action name `%s`. Rejecting as Forbidden, this is a programming error and a bug should be filed with as much information about the request that caused this as possible.', action)
            raise exception.Forbidden(message=_('Internal RBAC enforcement error, invalid rule (action) name.'))
        setattr(flask.g, _ENFORCEMENT_CHECK_ATTR, True)
        cls._assert_is_authenticated()
        if cls._shared_admin_auth_token_set():
            LOG.warning('RBAC: Bypassing authorization')
            return
        policy_dict.update(flask.request.view_args)
        if target_attr is None and build_target is None:
            try:
                policy_dict.update(cls._extract_member_target_data(member_target_type, member_target))
            except exception.NotFound:
                LOG.debug('Extracting inferred target data resulted in "NOT FOUND (404)".')
                raise
            except Exception as e:
                LOG.warning('Unable to extract inferred target data during enforcement')
                LOG.debug(e, exc_info=True)
                raise exception.ForbiddenAction(action=action)
            subj_token_target_data = cls._extract_subject_token_target_data()
            if subj_token_target_data:
                policy_dict.setdefault('target', {}).update(subj_token_target_data)
        else:
            if target_attr and build_target:
                raise ValueError('Programming Error: A target_attr or build_target must be provided, but not both')
            policy_dict['target'] = target_attr or build_target()
        json_input = flask.request.get_json(force=True, silent=True) or {}
        policy_dict.update(json_input.copy())
        policy_dict.update(cls._extract_filter_values(filters))
        flattened = utils.flatten_dict(policy_dict)
        if LOG.logger.getEffectiveLevel() <= log.DEBUG:
            args_str = ', '.join(['%s=%s' % (k, v) for k, v in (flask.request.view_args or {}).items()])
            args_str = strutils.mask_password(args_str)
            LOG.debug('RBAC: Authorizing `%(action)s(%(args)s)`', {'action': action, 'args': args_str})
        ctxt = cls._get_oslo_req_context()
        enforcer_obj = enforcer or cls()
        enforcer_obj._enforce(credentials=ctxt, action=action, target=flattened)
        LOG.debug('RBAC: Authorization granted')

    @classmethod
    def policy_enforcer_action(cls, action):
        """Decorator to set policy enforcement action name."""
        if action not in _POSSIBLE_TARGET_ACTIONS:
            raise ValueError('PROGRAMMING ERROR: Action must reference a valid Keystone policy enforcement name.')

        def wrapper(f):

            @functools.wraps(f)
            def inner(*args, **kwargs):
                setattr(flask.g, cls.ACTION_STORE_ATTR, action)
                return f(*args, **kwargs)
            return inner
        return wrapper

    @staticmethod
    def register_rules(enforcer):
        enforcer.register_defaults(policies.list_rules())