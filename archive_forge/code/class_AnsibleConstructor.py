from __future__ import (absolute_import, division, print_function)
from yaml.constructor import SafeConstructor, ConstructorError
from yaml.nodes import MappingNode
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.parsing.yaml.objects import AnsibleMapping, AnsibleSequence, AnsibleUnicode, AnsibleVaultEncryptedUnicode
from ansible.parsing.vault import VaultLib
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var
class AnsibleConstructor(SafeConstructor):

    def __init__(self, file_name=None, vault_secrets=None):
        self._ansible_file_name = file_name
        super(AnsibleConstructor, self).__init__()
        self._vaults = {}
        self.vault_secrets = vault_secrets or []
        self._vaults['default'] = VaultLib(secrets=self.vault_secrets)

    def construct_yaml_map(self, node):
        data = AnsibleMapping()
        yield data
        value = self.construct_mapping(node)
        data.update(value)
        data.ansible_pos = self._node_position_info(node)

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, 'expected a mapping node, but found %s' % node.id, node.start_mark)
        self.flatten_mapping(node)
        mapping = AnsibleMapping()
        mapping.ansible_pos = self._node_position_info(node)
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise ConstructorError('while constructing a mapping', node.start_mark, 'found unacceptable key (%s)' % exc, key_node.start_mark)
            if key in mapping:
                msg = u'While constructing a mapping from {1}, line {2}, column {3}, found a duplicate dict key ({0}). Using last defined value only.'.format(key, *mapping.ansible_pos)
                if C.DUPLICATE_YAML_DICT_KEY == 'warn':
                    display.warning(msg)
                elif C.DUPLICATE_YAML_DICT_KEY == 'error':
                    raise ConstructorError(context=None, context_mark=None, problem=to_native(msg), problem_mark=node.start_mark, note=None)
                else:
                    display.debug(msg)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

    def construct_yaml_str(self, node):
        value = self.construct_scalar(node)
        ret = AnsibleUnicode(value)
        ret.ansible_pos = self._node_position_info(node)
        return ret

    def construct_vault_encrypted_unicode(self, node):
        value = self.construct_scalar(node)
        b_ciphertext_data = to_bytes(value)
        vault = self._vaults['default']
        if vault.secrets is None:
            raise ConstructorError(context=None, context_mark=None, problem='found !vault but no vault password provided', problem_mark=node.start_mark, note=None)
        ret = AnsibleVaultEncryptedUnicode(b_ciphertext_data)
        ret.vault = vault
        ret.ansible_pos = self._node_position_info(node)
        return ret

    def construct_yaml_seq(self, node):
        data = AnsibleSequence()
        yield data
        data.extend(self.construct_sequence(node))
        data.ansible_pos = self._node_position_info(node)

    def construct_yaml_unsafe(self, node):
        try:
            constructor = getattr(node, 'id', 'object')
            if constructor is not None:
                constructor = getattr(self, 'construct_%s' % constructor)
        except AttributeError:
            constructor = self.construct_object
        value = constructor(node)
        return wrap_var(value)

    def _node_position_info(self, node):
        column = node.start_mark.column + 1
        line = node.start_mark.line + 1
        datasource = self._ansible_file_name or node.start_mark.name
        return (datasource, line, column)