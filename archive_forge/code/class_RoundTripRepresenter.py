from __future__ import print_function, absolute_import, division
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.compat import text_type, binary_type, to_unicode, PY2, PY3, ordereddict
from ruamel.yaml.compat import nprint, nprintf  # NOQA
from ruamel.yaml.scalarstring import (
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
import datetime
import sys
import types
from ruamel.yaml.comments import (
class RoundTripRepresenter(SafeRepresenter):

    def __init__(self, default_style=None, default_flow_style=None, dumper=None):
        if not hasattr(dumper, 'typ') and default_flow_style is None:
            default_flow_style = False
        SafeRepresenter.__init__(self, default_style=default_style, default_flow_style=default_flow_style, dumper=dumper)

    def ignore_aliases(self, data):
        try:
            if data.anchor is not None and data.anchor.value is not None:
                return False
        except AttributeError:
            pass
        return SafeRepresenter.ignore_aliases(self, data)

    def represent_none(self, data):
        if len(self.represented_objects) == 0 and (not self.serializer.use_explicit_start):
            return self.represent_scalar(u'tag:yaml.org,2002:null', u'null')
        return self.represent_scalar(u'tag:yaml.org,2002:null', '')

    def represent_literal_scalarstring(self, data):
        tag = None
        style = '|'
        anchor = data.yaml_anchor(any=True)
        if PY2 and (not isinstance(data, unicode)):
            data = unicode(data, 'ascii')
        tag = u'tag:yaml.org,2002:str'
        return self.represent_scalar(tag, data, style=style, anchor=anchor)
    represent_preserved_scalarstring = represent_literal_scalarstring

    def represent_folded_scalarstring(self, data):
        tag = None
        style = '>'
        anchor = data.yaml_anchor(any=True)
        for fold_pos in reversed(getattr(data, 'fold_pos', [])):
            if data[fold_pos] == ' ' and (fold_pos > 0 and (not data[fold_pos - 1].isspace())) and (fold_pos < len(data) and (not data[fold_pos + 1].isspace())):
                data = data[:fold_pos] + '\x07' + data[fold_pos:]
        if PY2 and (not isinstance(data, unicode)):
            data = unicode(data, 'ascii')
        tag = u'tag:yaml.org,2002:str'
        return self.represent_scalar(tag, data, style=style, anchor=anchor)

    def represent_single_quoted_scalarstring(self, data):
        tag = None
        style = "'"
        anchor = data.yaml_anchor(any=True)
        if PY2 and (not isinstance(data, unicode)):
            data = unicode(data, 'ascii')
        tag = u'tag:yaml.org,2002:str'
        return self.represent_scalar(tag, data, style=style, anchor=anchor)

    def represent_double_quoted_scalarstring(self, data):
        tag = None
        style = '"'
        anchor = data.yaml_anchor(any=True)
        if PY2 and (not isinstance(data, unicode)):
            data = unicode(data, 'ascii')
        tag = u'tag:yaml.org,2002:str'
        return self.represent_scalar(tag, data, style=style, anchor=anchor)

    def represent_plain_scalarstring(self, data):
        tag = None
        style = ''
        anchor = data.yaml_anchor(any=True)
        if PY2 and (not isinstance(data, unicode)):
            data = unicode(data, 'ascii')
        tag = u'tag:yaml.org,2002:str'
        return self.represent_scalar(tag, data, style=style, anchor=anchor)

    def insert_underscore(self, prefix, s, underscore, anchor=None):
        if underscore is None:
            return self.represent_scalar(u'tag:yaml.org,2002:int', prefix + s, anchor=anchor)
        if underscore[0]:
            sl = list(s)
            pos = len(s) - underscore[0]
            while pos > 0:
                sl.insert(pos, '_')
                pos -= underscore[0]
            s = ''.join(sl)
        if underscore[1]:
            s = '_' + s
        if underscore[2]:
            s += '_'
        return self.represent_scalar(u'tag:yaml.org,2002:int', prefix + s, anchor=anchor)

    def represent_scalar_int(self, data):
        if data._width is not None:
            s = '{:0{}d}'.format(data, data._width)
        else:
            s = format(data, 'd')
        anchor = data.yaml_anchor(any=True)
        return self.insert_underscore('', s, data._underscore, anchor=anchor)

    def represent_binary_int(self, data):
        if data._width is not None:
            s = '{:0{}b}'.format(data, data._width)
        else:
            s = format(data, 'b')
        anchor = data.yaml_anchor(any=True)
        return self.insert_underscore('0b', s, data._underscore, anchor=anchor)

    def represent_octal_int(self, data):
        if data._width is not None:
            s = '{:0{}o}'.format(data, data._width)
        else:
            s = format(data, 'o')
        anchor = data.yaml_anchor(any=True)
        return self.insert_underscore('0o', s, data._underscore, anchor=anchor)

    def represent_hex_int(self, data):
        if data._width is not None:
            s = '{:0{}x}'.format(data, data._width)
        else:
            s = format(data, 'x')
        anchor = data.yaml_anchor(any=True)
        return self.insert_underscore('0x', s, data._underscore, anchor=anchor)

    def represent_hex_caps_int(self, data):
        if data._width is not None:
            s = '{:0{}X}'.format(data, data._width)
        else:
            s = format(data, 'X')
        anchor = data.yaml_anchor(any=True)
        return self.insert_underscore('0x', s, data._underscore, anchor=anchor)

    def represent_scalar_float(self, data):
        """ this is way more complicated """
        value = None
        anchor = data.yaml_anchor(any=True)
        if data != data or (data == 0.0 and data == 1.0):
            value = u'.nan'
        elif data == self.inf_value:
            value = u'.inf'
        elif data == -self.inf_value:
            value = u'-.inf'
        if value:
            return self.represent_scalar(u'tag:yaml.org,2002:float', value, anchor=anchor)
        if data._exp is None and data._prec > 0 and (data._prec == data._width - 1):
            value = u'{}{:d}.'.format(data._m_sign if data._m_sign else '', abs(int(data)))
        elif data._exp is None:
            prec = data._prec
            ms = data._m_sign if data._m_sign else ''
            value = u'{}{:0{}.{}f}'.format(ms, abs(data), data._width - len(ms), data._width - prec - 1)
            if prec == 0 or (prec == 1 and ms != ''):
                value = value.replace(u'0.', u'.')
            while len(value) < data._width:
                value += u'0'
        else:
            m, es = u'{:{}.{}e}'.format(data, data._width, data._width + (1 if data._m_sign else 0)).split('e')
            w = data._width if data._prec > 0 else data._width + 1
            if data < 0:
                w += 1
            m = m[:w]
            e = int(es)
            m1, m2 = m.split('.')
            while len(m1) + len(m2) < data._width - (1 if data._prec >= 0 else 0):
                m2 += u'0'
            if data._m_sign and data > 0:
                m1 = '+' + m1
            esgn = u'+' if data._e_sign else ''
            if data._prec < 0:
                if m2 != u'0':
                    e -= len(m2)
                else:
                    m2 = ''
                while len(m1) + len(m2) - (1 if data._m_sign else 0) < data._width:
                    m2 += u'0'
                    e -= 1
                value = m1 + m2 + data._exp + u'{:{}0{}d}'.format(e, esgn, data._e_width)
            elif data._prec == 0:
                e -= len(m2)
                value = m1 + m2 + u'.' + data._exp + u'{:{}0{}d}'.format(e, esgn, data._e_width)
            else:
                if data._m_lead0 > 0:
                    m2 = u'0' * (data._m_lead0 - 1) + m1 + m2
                    m1 = u'0'
                    m2 = m2[:-data._m_lead0]
                    e += data._m_lead0
                while len(m1) < data._prec:
                    m1 += m2[0]
                    m2 = m2[1:]
                    e -= 1
                value = m1 + u'.' + m2 + data._exp + u'{:{}0{}d}'.format(e, esgn, data._e_width)
        if value is None:
            value = to_unicode(repr(data)).lower()
        return self.represent_scalar(u'tag:yaml.org,2002:float', value, anchor=anchor)

    def represent_sequence(self, tag, sequence, flow_style=None):
        value = []
        try:
            flow_style = sequence.fa.flow_style(flow_style)
        except AttributeError:
            flow_style = flow_style
        try:
            anchor = sequence.yaml_anchor()
        except AttributeError:
            anchor = None
        node = SequenceNode(tag, value, flow_style=flow_style, anchor=anchor)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        best_style = True
        try:
            comment = getattr(sequence, comment_attrib)
            node.comment = comment.comment
            if node.comment and node.comment[1]:
                for ct in node.comment[1]:
                    ct.reset()
            item_comments = comment.items
            for v in item_comments.values():
                if v and v[1]:
                    for ct in v[1]:
                        ct.reset()
            item_comments = comment.items
            node.comment = comment.comment
            try:
                node.comment.append(comment.end)
            except AttributeError:
                pass
        except AttributeError:
            item_comments = {}
        for idx, item in enumerate(sequence):
            node_item = self.represent_data(item)
            self.merge_comments(node_item, item_comments.get(idx))
            if not (isinstance(node_item, ScalarNode) and (not node_item.style)):
                best_style = False
            value.append(node_item)
        if flow_style is None:
            if len(sequence) != 0 and self.default_flow_style is not None:
                node.flow_style = self.default_flow_style
            else:
                node.flow_style = best_style
        return node

    def merge_comments(self, node, comments):
        if comments is None:
            assert hasattr(node, 'comment')
            return node
        if getattr(node, 'comment', None) is not None:
            for idx, val in enumerate(comments):
                if idx >= len(node.comment):
                    continue
                nc = node.comment[idx]
                if nc is not None:
                    assert val is None or val == nc
                    comments[idx] = nc
        node.comment = comments
        return node

    def represent_key(self, data):
        if isinstance(data, CommentedKeySeq):
            self.alias_key = None
            return self.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)
        if isinstance(data, CommentedKeyMap):
            self.alias_key = None
            return self.represent_mapping(u'tag:yaml.org,2002:map', data, flow_style=True)
        return SafeRepresenter.represent_key(self, data)

    def represent_mapping(self, tag, mapping, flow_style=None):
        value = []
        try:
            flow_style = mapping.fa.flow_style(flow_style)
        except AttributeError:
            flow_style = flow_style
        try:
            anchor = mapping.yaml_anchor()
        except AttributeError:
            anchor = None
        node = MappingNode(tag, value, flow_style=flow_style, anchor=anchor)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        best_style = True
        try:
            comment = getattr(mapping, comment_attrib)
            node.comment = comment.comment
            if node.comment and node.comment[1]:
                for ct in node.comment[1]:
                    ct.reset()
            item_comments = comment.items
            for v in item_comments.values():
                if v and v[1]:
                    for ct in v[1]:
                        ct.reset()
            try:
                node.comment.append(comment.end)
            except AttributeError:
                pass
        except AttributeError:
            item_comments = {}
        merge_list = [m[1] for m in getattr(mapping, merge_attrib, [])]
        try:
            merge_pos = getattr(mapping, merge_attrib, [[0]])[0][0]
        except IndexError:
            merge_pos = 0
        item_count = 0
        if bool(merge_list):
            items = mapping.non_merged_items()
        else:
            items = mapping.items()
        for item_key, item_value in items:
            item_count += 1
            node_key = self.represent_key(item_key)
            node_value = self.represent_data(item_value)
            item_comment = item_comments.get(item_key)
            if item_comment:
                assert getattr(node_key, 'comment', None) is None
                node_key.comment = item_comment[:2]
                nvc = getattr(node_value, 'comment', None)
                if nvc is not None:
                    nvc[0] = item_comment[2]
                    nvc[1] = item_comment[3]
                else:
                    node_value.comment = item_comment[2:]
            if not (isinstance(node_key, ScalarNode) and (not node_key.style)):
                best_style = False
            if not (isinstance(node_value, ScalarNode) and (not node_value.style)):
                best_style = False
            value.append((node_key, node_value))
        if flow_style is None:
            if (item_count != 0 or bool(merge_list)) and self.default_flow_style is not None:
                node.flow_style = self.default_flow_style
            else:
                node.flow_style = best_style
        if bool(merge_list):
            if len(merge_list) == 1:
                arg = self.represent_data(merge_list[0])
            else:
                arg = self.represent_data(merge_list)
                arg.flow_style = True
            value.insert(merge_pos, (ScalarNode(u'tag:yaml.org,2002:merge', '<<'), arg))
        return node

    def represent_omap(self, tag, omap, flow_style=None):
        value = []
        try:
            flow_style = omap.fa.flow_style(flow_style)
        except AttributeError:
            flow_style = flow_style
        try:
            anchor = omap.yaml_anchor()
        except AttributeError:
            anchor = None
        node = SequenceNode(tag, value, flow_style=flow_style, anchor=anchor)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        best_style = True
        try:
            comment = getattr(omap, comment_attrib)
            node.comment = comment.comment
            if node.comment and node.comment[1]:
                for ct in node.comment[1]:
                    ct.reset()
            item_comments = comment.items
            for v in item_comments.values():
                if v and v[1]:
                    for ct in v[1]:
                        ct.reset()
            try:
                node.comment.append(comment.end)
            except AttributeError:
                pass
        except AttributeError:
            item_comments = {}
        for item_key in omap:
            item_val = omap[item_key]
            node_item = self.represent_data({item_key: item_val})
            item_comment = item_comments.get(item_key)
            if item_comment:
                if item_comment[1]:
                    node_item.comment = [None, item_comment[1]]
                assert getattr(node_item.value[0][0], 'comment', None) is None
                node_item.value[0][0].comment = [item_comment[0], None]
                nvc = getattr(node_item.value[0][1], 'comment', None)
                if nvc is not None:
                    nvc[0] = item_comment[2]
                    nvc[1] = item_comment[3]
                else:
                    node_item.value[0][1].comment = item_comment[2:]
            value.append(node_item)
        if flow_style is None:
            if self.default_flow_style is not None:
                node.flow_style = self.default_flow_style
            else:
                node.flow_style = best_style
        return node

    def represent_set(self, setting):
        flow_style = False
        tag = u'tag:yaml.org,2002:set'
        value = []
        flow_style = setting.fa.flow_style(flow_style)
        try:
            anchor = setting.yaml_anchor()
        except AttributeError:
            anchor = None
        node = MappingNode(tag, value, flow_style=flow_style, anchor=anchor)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        best_style = True
        try:
            comment = getattr(setting, comment_attrib)
            node.comment = comment.comment
            if node.comment and node.comment[1]:
                for ct in node.comment[1]:
                    ct.reset()
            item_comments = comment.items
            for v in item_comments.values():
                if v and v[1]:
                    for ct in v[1]:
                        ct.reset()
            try:
                node.comment.append(comment.end)
            except AttributeError:
                pass
        except AttributeError:
            item_comments = {}
        for item_key in setting.odict:
            node_key = self.represent_key(item_key)
            node_value = self.represent_data(None)
            item_comment = item_comments.get(item_key)
            if item_comment:
                assert getattr(node_key, 'comment', None) is None
                node_key.comment = item_comment[:2]
            node_key.style = node_value.style = '?'
            if not (isinstance(node_key, ScalarNode) and (not node_key.style)):
                best_style = False
            if not (isinstance(node_value, ScalarNode) and (not node_value.style)):
                best_style = False
            value.append((node_key, node_value))
        best_style = best_style
        return node

    def represent_dict(self, data):
        """write out tag if saved on loading"""
        try:
            t = data.tag.value
        except AttributeError:
            t = None
        if t:
            if t.startswith('!!'):
                tag = 'tag:yaml.org,2002:' + t[2:]
            else:
                tag = t
        else:
            tag = u'tag:yaml.org,2002:map'
        return self.represent_mapping(tag, data)

    def represent_list(self, data):
        try:
            t = data.tag.value
        except AttributeError:
            t = None
        if t:
            if t.startswith('!!'):
                tag = 'tag:yaml.org,2002:' + t[2:]
            else:
                tag = t
        else:
            tag = u'tag:yaml.org,2002:seq'
        return self.represent_sequence(tag, data)

    def represent_datetime(self, data):
        inter = 'T' if data._yaml['t'] else ' '
        _yaml = data._yaml
        if _yaml['delta']:
            data += _yaml['delta']
            value = data.isoformat(inter)
        else:
            value = data.isoformat(inter)
        if _yaml['tz']:
            value += _yaml['tz']
        return self.represent_scalar(u'tag:yaml.org,2002:timestamp', to_unicode(value))

    def represent_tagged_scalar(self, data):
        try:
            tag = data.tag.value
        except AttributeError:
            tag = None
        try:
            anchor = data.yaml_anchor()
        except AttributeError:
            anchor = None
        return self.represent_scalar(tag, data.value, style=data.style, anchor=anchor)

    def represent_scalar_bool(self, data):
        try:
            anchor = data.yaml_anchor()
        except AttributeError:
            anchor = None
        return SafeRepresenter.represent_bool(self, data, anchor=anchor)