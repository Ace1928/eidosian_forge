from importlib import import_module
import os
import sys
class AMap:

    def __init__(self, head, tail, indent=4 * ' '):
        self.mod = load(head, tail)
        self.variable = {}
        self.vars = []
        self.text = []
        self.indent = indent
        for key, val in self.mod.__dict__.items():
            if key.startswith('__'):
                continue
            elif key == 'MAP':
                continue
            else:
                self.variable[key] = val
                self.vars.append(key)
        self.vars.sort()

    def sync(self):
        for key, val in self.mod.MAP['fro'].items():
            try:
                assert self.mod.MAP['to'][val] == key
            except KeyError:
                print(f'# Added {self.mod.MAP['to'][val]}={key}')
                self.mod.MAP['to'][val] = key
            except AssertionError:
                raise Exception(f"Mismatch key:{key} '{val}' != '{self.mod.MAP['to'][val]}'")
        for val in self.mod.MAP['to'].values():
            if val not in self.mod.MAP['fro']:
                print(f"# Missing URN '{val}'")

    def do_fro(self):
        txt = ["%s'fro': {" % self.indent]
        i2 = self.indent + self.indent
        _fro = self.mod.MAP['fro']
        for var in self.vars:
            _v = self.variable[var]
            li = [k[len(_v):] for k in _fro.keys() if k.startswith(_v)]
            li.sort(intcmp)
            for item in li:
                txt.append(f"{i2}{var}+'{item}': '{_fro[_v + item]}',")
        txt.append('%s},' % self.indent)
        return txt

    def do_to(self):
        txt = ["%s'to': {" % self.indent]
        i2 = self.indent + self.indent
        _to = self.mod.MAP['to']
        _keys = _to.keys()
        _keys.sort()
        invmap = {v: k for k, v in self.variable.items()}
        for key in _keys:
            val = _to[key]
            for _urn, _name in invmap.items():
                if val.startswith(_urn):
                    txt.append(f"{i2}'{key}': {_name}+'{val[len(_urn):]}',")
        txt.append('%s}' % self.indent)
        return txt

    def __str__(self):
        self.sync()
        text = []
        for key in self.vars:
            text.append(f"{key} = '{self.variable[key]}'")
        text.extend(['', ''])
        text.append('MAP = {')
        text.append(f"{self.indent}'identifier': '{self.mod.MAP['identifier']}',")
        text.extend(self.do_fro())
        text.extend(self.do_to())
        text.append('}')
        text.append('')
        return '\n'.join(text)