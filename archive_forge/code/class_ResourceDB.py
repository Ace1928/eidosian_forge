import functools
import re
import sys
from Xlib.support import lock
class ResourceDB:

    def __init__(self, file=None, string=None, resources=None):
        self.db = {}
        self.lock = lock.allocate_lock()
        if file is not None:
            self.insert_file(file)
        if string is not None:
            self.insert_string(string)
        if resources is not None:
            self.insert_resources(resources)

    def insert_file(self, file):
        """insert_file(file)

        Load resources entries from FILE, and insert them into the
        database.  FILE can be a filename (a string)or a file object.

        """
        if type(file) is str:
            file = open(file, 'r')
        self.insert_string(file.read())

    def insert_string(self, data):
        """insert_string(data)

        Insert the resources entries in the string DATA into the
        database.

        """
        lines = data.split('\n')
        while lines:
            line = lines[0]
            del lines[0]
            if not line:
                continue
            if comment_re.match(line):
                continue
            while line[-1] == '\\':
                if lines:
                    line = line[:-1] + lines[0]
                    del lines[0]
                else:
                    line = line[:-1]
                    break
            m = resource_spec_re.match(line)
            if not m:
                continue
            res, value = m.group(1, 2)
            splits = value_escape_re.split(value)
            for i in range(1, len(splits), 2):
                s = splits[i]
                if len(s) == 3:
                    splits[i] = chr(int(s, 8))
                elif s == 'n':
                    splits[i] = '\n'
            splits[-1] = splits[-1].rstrip()
            value = ''.join(splits)
            self.insert(res, value)

    def insert_resources(self, resources):
        """insert_resources(resources)

        Insert all resources entries in the list RESOURCES into the
        database.  Each element in RESOURCES should be a tuple:

          (resource, value)

        Where RESOURCE is a string and VALUE can be any Python value.

        """
        for res, value in resources:
            self.insert(res, value)

    def insert(self, resource, value):
        """insert(resource, value)

        Insert a resource entry into the database.  RESOURCE is a
        string and VALUE can be any Python value.

        """
        parts = resource_parts_re.split(resource)
        if parts[-1] == '':
            return
        self.lock.acquire()
        db = self.db
        for i in range(1, len(parts), 2):
            if parts[i - 1] not in db:
                db[parts[i - 1]] = ({}, {})
            if '*' in parts[i]:
                db = db[parts[i - 1]][1]
            else:
                db = db[parts[i - 1]][0]
        if parts[-1] in db:
            db[parts[-1]] = db[parts[-1]][:2] + (value,)
        else:
            db[parts[-1]] = ({}, {}, value)
        self.lock.release()

    def __getitem__(self, nc):
        """db[name, class]

        Return the value matching the resource identified by NAME and
        CLASS.  If no match is found, KeyError is raised.
        """
        name, cls = nc
        namep = name.split('.')
        clsp = cls.split('.')
        if len(namep) != len(clsp):
            raise ValueError('Different number of parts in resource name/class: %s/%s' % (name, cls))
        complen = len(namep)
        matches = []
        self.lock.acquire()
        try:
            if namep[0] in self.db:
                bin_insert(matches, _Match((NAME_MATCH,), self.db[namep[0]]))
            if clsp[0] in self.db:
                bin_insert(matches, _Match((CLASS_MATCH,), self.db[clsp[0]]))
            if '?' in self.db:
                bin_insert(matches, _Match((WILD_MATCH,), self.db['?']))
            if complen == 1 and matches:
                x = matches[0]
                if x.final(complen):
                    return x.value()
                else:
                    raise KeyError((name, cls))
            if '' in self.db:
                bin_insert(matches, _Match((), self.db[''][1]))
            while matches:
                x = matches[0]
                del matches[0]
                i = x.match_length()
                for part, score in ((namep[i], NAME_MATCH), (clsp[i], CLASS_MATCH), ('?', WILD_MATCH)):
                    match = x.match(part, score)
                    if match:
                        if match.final(complen):
                            return match.value()
                        else:
                            bin_insert(matches, match)
                    match = x.skip_match(complen)
                    if match:
                        bin_insert(matches, match)
            raise KeyError((name, cls))
        finally:
            self.lock.release()

    def get(self, res, cls, default=None):
        """get(name, class [, default])

        Return the value matching the resource identified by NAME and
        CLASS.  If no match is found, DEFAULT is returned, or None if
        DEFAULT isn't specified.

        """
        try:
            return self[res, cls]
        except KeyError:
            return default

    def update(self, db):
        """update(db)

        Update this database with all resources entries in the resource
        database DB.

        """
        self.lock.acquire()
        update_db(self.db, db.db)
        self.lock.release()

    def output(self):
        """output()

        Return the resource database in text representation.
        """
        self.lock.acquire()
        text = output_db('', self.db)
        self.lock.release()
        return text

    def getopt(self, name, argv, opts):
        """getopt(name, argv, opts)

        Parse X command line options, inserting the recognised options
        into the resource database.

        NAME is the application name, and will be prepended to all
        specifiers.  ARGV is the list of command line arguments,
        typically sys.argv[1:].

        OPTS is a mapping of options to resource specifiers.  The key is
        the option flag (with leading -), and the value is an instance of
        some Option subclass:

        NoArg(specifier, value): set resource to value.
        IsArg(specifier):        set resource to option itself
        SepArg(specifier):       value is next argument
        ResArg:                  resource and value in next argument
        SkipArg:                 ignore this option and next argument
        SkipLine:                ignore rest of arguments
        SkipNArgs(count):        ignore this option and count arguments

        The remaining, non-option, oparguments is returned.

        rdb.OptionError is raised if there is an error in the argument list.
        """
        while argv and argv[0] and (argv[0][0] == '-'):
            try:
                argv = opts[argv[0]].parse(name, self, argv)
            except KeyError:
                raise OptionError('unknown option: %s' % argv[0])
            except IndexError:
                raise OptionError('missing argument to option: %s' % argv[0])
        return argv