import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
class V4toV5(Adapter):
    """Convert msg spec V4 to V5"""
    version = '5.0'
    msg_type_map = {v: k for k, v in V5toV4.msg_type_map.items()}

    def update_header(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Update the header."""
        msg['header']['version'] = self.version
        if msg['parent_header']:
            msg['parent_header']['version'] = self.version
        return msg

    def kernel_info_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a kernel info reply."""
        content = msg['content']
        for key in ('protocol_version', 'ipython_version'):
            if key in content:
                content[key] = '.'.join(map(str, content[key]))
        content.setdefault('protocol_version', '4.1')
        if content['language'].startswith('python') and 'ipython_version' in content:
            content['implementation'] = 'ipython'
            content['implementation_version'] = content.pop('ipython_version')
        language = content.pop('language')
        language_info = content.setdefault('language_info', {})
        language_info.setdefault('name', language)
        if 'language_version' in content:
            language_version = '.'.join(map(str, content.pop('language_version')))
            language_info.setdefault('version', language_version)
        content['banner'] = ''
        return msg

    def execute_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an execute request."""
        content = msg['content']
        user_variables = content.pop('user_variables', [])
        user_expressions = content.setdefault('user_expressions', {})
        for v in user_variables:
            user_expressions[v] = v
        return msg

    def execute_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an execute reply."""
        content = msg['content']
        user_expressions = content.setdefault('user_expressions', {})
        user_variables = content.pop('user_variables', {})
        if user_variables:
            user_expressions.update(user_variables)
        for payload in content.get('payload', []):
            if payload.get('source', None) == 'page' and 'text' in payload:
                if 'data' not in payload:
                    payload['data'] = {}
                payload['data']['text/plain'] = payload.pop('text')
        return msg

    def complete_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a complete request."""
        old_content = msg['content']
        new_content = msg['content'] = {}
        new_content['code'] = old_content['line']
        new_content['cursor_pos'] = old_content['cursor_pos']
        return msg

    def complete_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a complete reply."""
        content = msg['content']
        new_content = msg['content'] = {'status': 'ok'}
        new_content['matches'] = content['matches']
        if content['matched_text']:
            new_content['cursor_start'] = -len(content['matched_text'])
        else:
            new_content['cursor_start'] = None
        new_content['cursor_end'] = None
        new_content['metadata'] = {}
        return msg

    def inspect_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an inspect request."""
        content = msg['content']
        name = content['oname']
        new_content = msg['content'] = {}
        new_content['code'] = name
        new_content['cursor_pos'] = len(name)
        new_content['detail_level'] = content['detail_level']
        return msg

    def inspect_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """inspect_reply can't be easily backward compatible"""
        content = msg['content']
        new_content = msg['content'] = {'status': 'ok'}
        found = new_content['found'] = content['found']
        new_content['data'] = data = {}
        new_content['metadata'] = {}
        if found:
            lines = []
            for key in ('call_def', 'init_definition', 'definition'):
                if content.get(key, False):
                    lines.append(content[key])
                    break
            for key in ('call_docstring', 'init_docstring', 'docstring'):
                if content.get(key, False):
                    lines.append(content[key])
                    break
            if not lines:
                lines.append('<empty docstring>')
            data['text/plain'] = '\n'.join(lines)
        return msg

    def stream(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a stream message."""
        content = msg['content']
        content['text'] = content.pop('data')
        return msg

    def display_data(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle display data."""
        content = msg['content']
        content.pop('source', None)
        data = content['data']
        if 'application/json' in data:
            try:
                data['application/json'] = json.loads(data['application/json'])
            except Exception:
                pass
        return msg

    def input_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an input request."""
        msg['content'].setdefault('password', False)
        return msg