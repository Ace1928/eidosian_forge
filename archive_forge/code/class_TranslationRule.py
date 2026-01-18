import functools
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import function
@functools.total_ordering
class TranslationRule(object):
    """Translating mechanism one properties to another.

    Mechanism uses list of rules, each defines by this class, and can be
    executed. Working principe: during resource creating after properties
    defining resource take list of rules, specified by method
    translation_rules, which should be overloaded for each resource, if it's
    needed, and execute each rule using translate_properties method. Next
    operations are allowed:

    - ADD. This rule allows to add some value to list-type properties. Only
           list-type values can be added to such properties. Using for other
           cases is prohibited and will be returned with error.
    - REPLACE. This rule allows to replace some property value to another. Used
           for all types of properties. Note, that if property has list type,
           then value will be replaced for all elements of list, where it
           needed. If element in such property must be replaced by value of
           another element of this property, value_name must be defined.
    - DELETE. This rule allows to delete some property. If property has list
           type, then deleting affects value in all list elements.
    - RESOLVE. This rule allows to resolve some property using client and
           the finder function. Finders may require an additional entity key.
    """
    RULE_KEYS = ADD, REPLACE, DELETE, RESOLVE = ('Add', 'Replace', 'Delete', 'Resolve')

    def __lt__(self, other):
        rules = [TranslationRule.ADD, TranslationRule.REPLACE, TranslationRule.RESOLVE, TranslationRule.DELETE]
        idx1 = rules.index(self.rule)
        idx2 = rules.index(other.rule)
        return idx1 < idx2

    def __init__(self, properties, rule, translation_path, value=None, value_name=None, value_path=None, client_plugin=None, finder=None, entity=None, custom_value_path=None):
        """Add new rule for translating mechanism.

        :param properties: properties of resource
        :param rule: rule from RULE_KEYS
        :param translation_path: list with path to property, which value will
               be affected in rule.
        :param value: value which will be involved in rule
        :param value_name: value_name which used for replacing properties
               inside list-type properties.
        :param value_path: path to value, which should be used for translation.
        :param client_plugin: client plugin that would be used to resolve
        :param finder: finder method of the client plugin
        :param entity: some generic finders require an entity to resolve ex.
               neutron finder function.
        :param custom_value_path: list-type value path to translate property,
               which has no schema.
        """
        self.properties = properties
        self.rule = rule
        self.translation_path = translation_path
        self.value = value or None
        self.value_name = value_name
        self.value_path = value_path
        self.client_plugin = client_plugin
        self.finder = finder
        self.entity = entity
        self.custom_value_path = custom_value_path
        self.validate()

    def validate(self):
        if self.rule not in self.RULE_KEYS:
            raise ValueError(_('There is no rule %(rule)s. List of allowed rules is: %(rules)s.') % {'rule': self.rule, 'rules': ', '.join(self.RULE_KEYS)})
        if not isinstance(self.translation_path, list) or len(self.translation_path) == 0:
            raise ValueError(_('"translation_path" should be non-empty list with path to translate.'))
        args = [self.value_path is not None, self.value is not None, self.value_name is not None]
        if args.count(True) > 1:
            raise ValueError(_('"value_path", "value" and "value_name" are mutually exclusive and cannot be specified at the same time.'))
        if self.rule == self.ADD and self.value is not None and (not isinstance(self.value, list)):
            raise ValueError(_('"value" must be list type when rule is Add.'))
        if self.rule == self.RESOLVE and (not (self.client_plugin or self.finder)):
            raise ValueError(_('"client_plugin" and "finder" should be specified for %s rule') % self.RESOLVE)

    def get_value_absolute_path(self, full_value_name=False):
        path = []
        if self.value_name:
            if full_value_name:
                path.extend(self.translation_path[:-1])
            path.append(self.value_name)
        elif self.value_path:
            path.extend(self.value_path)
        if self.custom_value_path:
            path.extend(self.custom_value_path)
        return path