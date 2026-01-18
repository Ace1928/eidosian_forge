class IniConfig(object):

    def __init__(self, path, data=None):
        self.path = str(path)
        if data is None:
            f = open(self.path)
            try:
                tokens = self._parse(iter(f))
            finally:
                f.close()
        else:
            tokens = self._parse(data.splitlines(True))
        self._sources = {}
        self.sections = {}
        for lineno, section, name, value in tokens:
            if section is None:
                self._raise(lineno, 'no section header defined')
            self._sources[section, name] = lineno
            if name is None:
                if section in self.sections:
                    self._raise(lineno, 'duplicate section %r' % (section,))
                self.sections[section] = {}
            else:
                if name in self.sections[section]:
                    self._raise(lineno, 'duplicate name %r' % (name,))
                self.sections[section][name] = value

    def _raise(self, lineno, msg):
        raise ParseError(self.path, lineno, msg)

    def _parse(self, line_iter):
        result = []
        section = None
        for lineno, line in enumerate(line_iter):
            name, data = self._parseline(line, lineno)
            if name is not None and data is not None:
                result.append((lineno, section, name, data))
            elif name is not None and data is None:
                if not name:
                    self._raise(lineno, 'empty section name')
                section = name
                result.append((lineno, section, None, None))
            elif name is None and data is not None:
                if not result:
                    self._raise(lineno, 'unexpected value continuation')
                last = result.pop()
                last_name, last_data = last[-2:]
                if last_name is None:
                    self._raise(lineno, 'unexpected value continuation')
                if last_data:
                    data = '%s\n%s' % (last_data, data)
                result.append(last[:-1] + (data,))
        return result

    def _parseline(self, line, lineno):
        if iscommentline(line):
            line = ''
        else:
            line = line.rstrip()
        if not line:
            return (None, None)
        if line[0] == '[':
            realline = line
            for c in COMMENTCHARS:
                line = line.split(c)[0].rstrip()
            if line[-1] == ']':
                return (line[1:-1], None)
            return (None, realline.strip())
        elif not line[0].isspace():
            try:
                name, value = line.split('=', 1)
                if ':' in name:
                    raise ValueError()
            except ValueError:
                try:
                    name, value = line.split(':', 1)
                except ValueError:
                    self._raise(lineno, 'unexpected line: %r' % line)
            return (name.strip(), value.strip())
        else:
            return (None, line.strip())

    def lineof(self, section, name=None):
        lineno = self._sources.get((section, name))
        if lineno is not None:
            return lineno + 1

    def get(self, section, name, default=None, convert=str):
        try:
            return convert(self.sections[section][name])
        except KeyError:
            return default

    def __getitem__(self, name):
        if name not in self.sections:
            raise KeyError(name)
        return SectionWrapper(self, name)

    def __iter__(self):
        for name in sorted(self.sections, key=self.lineof):
            yield SectionWrapper(self, name)

    def __contains__(self, arg):
        return arg in self.sections