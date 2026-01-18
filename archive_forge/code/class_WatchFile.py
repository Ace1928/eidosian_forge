import logging
import re
class WatchFile(object):
    """A Debian watch file.

    :ivar entries: list of Watch entries
    :ivar options: optional list of global options, applied to all Watch
        entries
    :ivar version: watch file version
    """

    def __init__(self, entries=None, options=None, version=DEFAULT_VERSION):
        self.version = version
        if entries is None:
            entries = []
        self.entries = entries
        if options is None:
            options = []
        self.options = options

    def __iter__(self):
        return iter(self.entries)

    def dump(self, f):
        """Write the contents of a watch file to a file-like object.

        Note that this will not preserve the formatting of the original file,
        and thus it is currently not possible to use this function to
        parse and reserialize a file and end up with the same contents.

        :param f: File-like object to write to
        """

        def serialize_options(opts):
            s = ','.join(opts)
            if ' ' in s or '\t' in s:
                return 'opts="' + s + '"'
            return 'opts=' + s
        if self.version is not None:
            f.write('version=%d\n' % self.version)
        if self.options:
            f.write(serialize_options(self.options) + '\n')
        for entry in self.entries:
            if entry.options:
                f.write(serialize_options(entry.options) + ' ')
            f.write(entry.url)
            if entry.matching_pattern:
                f.write(' ' + entry.matching_pattern)
            if entry.version:
                f.write(' ' + entry.version)
            if entry.script:
                f.write(' ' + entry.script)
            f.write('\n')

    @classmethod
    def from_lines(cls, lines, strict=False):
        """Parse from the contents that make up a watch file.

        :param lines: watch file lines to parse
        :return: instance or None if there are no non-comment lines in the file
        :raise MissingVersion: if there is no version number declared
        :raise ValueError: when syntax errors are encountered
        """
        joined_lines = []
        continued = []
        for line in lines:
            if line.startswith('#'):
                continue
            if not line.strip():
                continue
            if line.rstrip('\n').endswith('\\'):
                continued.append(line.rstrip('\n\\'))
            else:
                continued.append(line)
                joined_lines.append(continued)
                continued = []
        if continued:
            _complain('watchfile ended with \\; skipping last line', strict)
            joined_lines.append(continued)
        if not joined_lines:
            return None
        firstline = ''.join(joined_lines.pop(0))
        try:
            key, value = firstline.split('=', 1)
        except ValueError:
            raise MissingVersion()
        if key.strip() != 'version':
            raise MissingVersion()
        version = int(value.strip())
        persistent_options = []
        entries = []
        for chunked in joined_lines:
            if version > 3:
                chunked = [chunk.lstrip() for chunk in chunked]
            line = ''.join(chunked).strip()
            if not line:
                continue
            if line.startswith('opts='):
                if line[5] == '"':
                    optend = line.index('"', 6)
                    if optend == -1:
                        raise ValueError('Not matching " in %r' % line)
                    opts_str = line[6:optend]
                    line = line[optend + 1:]
                else:
                    try:
                        opts_str, line = line[5:].split(None, 1)
                    except ValueError:
                        opts_str = line[5:]
                        line = ''
                opts = opts_str.split(',')
            else:
                opts = []
            if line:
                try:
                    url, line = line.split(None, 1)
                except ValueError:
                    url = line
                    line = ''
                m = re.findall('/([^/]*\\([^/]*\\)[^/]*)$', url)
                if m:
                    parts = (str(m[0]),) + tuple(line.split(None, 1))
                    url = url[:-len(m[0]) - 1]
                else:
                    parts = tuple(line.split(None, 2))
                entries.append(Watch(url, *parts, opts=opts))
            else:
                persistent_options.extend(opts)
        return cls(entries=entries, options=persistent_options, version=version)