from pathlib import Path
import sys
class ArFile(object):
    """ Representation of an ar archive, see man 1 ar.

    The interface of this class tries to mimic that of the TarFile module in
    the standard library.

    ArFile objects have the following (read-only) properties:
        - members       same as getmembers()
    """

    def __init__(self, filename=None, mode='r', fileobj=None, encoding=None, errors=None):
        """ Build an ar file representation starting from either a filename or
        an existing file object. The only supported mode is 'r'.

        The encoding and errors parameters control how member
        names are decoded into Unicode strings. Like tarfile, the default
        encoding is sys.getfilesystemencoding() and the default error handling
        scheme is 'surrogateescape'.
        """
        self.__members = []
        self.__members_dict = {}
        self.__fname = filename
        self.__fileobj = fileobj
        self.__encoding = encoding or sys.getfilesystemencoding()
        if errors is None:
            errors = 'surrogateescape'
        self.__errors = errors
        if mode == 'r':
            self.__index_archive()

    def __index_archive(self):
        if self.__fname:
            with open(self.__fname, 'rb') as fp:
                self.__collect_members(fp)
        elif self.__fileobj:
            self.__collect_members(self.__fileobj)
        else:
            raise ArError('Unable to open valid file')

    def __collect_members(self, fp):
        if fp.read(GLOBAL_HEADER_LENGTH) != GLOBAL_HEADER:
            raise ArError('Unable to find global header')
        while True:
            newmember = ArMember.from_file(fp, self.__fname, encoding=self.__encoding, errors=self.__errors)
            if not newmember:
                break
            self.__members.append(newmember)
            self.__members_dict[newmember.name] = newmember
            if newmember.size % 2 == 0:
                fp.seek(newmember.size, 1)
            else:
                fp.seek(newmember.size + 1, 1)

    def getmember(self, name):
        """ Return the (last occurrence of a) member in the archive whose name
        is 'name'. Raise KeyError if no member matches the given name.

        Note that in case of name collisions the only way to retrieve all
        members matching a given name is to use getmembers. """
        return self.__members_dict[name]

    def getmembers(self):
        """ Return a list of all members contained in the archive.

        The list has the same order of members in the archive and can contain
        duplicate members (i.e. members with the same name) if they are
        duplicate in the archive itself. """
        return self.__members
    members = property(getmembers)

    def getnames(self):
        """ Return a list of all member names in the archive. """
        return [f.name for f in self.__members]

    def extractall(self):
        """ Not (yet) implemented. """
        raise NotImplementedError

    def extract(self, member, path):
        """ Not (yet) implemented. """
        raise NotImplementedError

    def extractfile(self, member):
        """ Return a file object corresponding to the requested member. A member
        can be specified either as a string (its name) or as a ArMember
        instance. """
        for m in self.__members:
            if isinstance(member, ArMember) and m.name == member.name:
                return m
            if member == m.name:
                return m
        return None

    def __iter__(self):
        """ Iterate over the members of the present ar archive. """
        return iter(self.__members)

    def __getitem__(self, name):
        """ Same as .getmember(name). """
        return self.getmember(name)