from pygame.base import get_sdl_version
class SoftwareVersion(tuple):
    """
    A class for storing data about software versions.
    """
    __slots__ = ()
    fields = ('major', 'minor', 'patch')

    def __new__(cls, major, minor, patch):
        return tuple.__new__(cls, (major, minor, patch))

    def __repr__(self):
        fields = (f'{fld}={val}' for fld, val in zip(self.fields, self))
        return f'{str(self.__class__.__name__)}({', '.join(fields)})'

    def __str__(self):
        return f'{self.major}.{self.minor}.{self.patch}'
    major = property(lambda self: self[0])
    minor = property(lambda self: self[1])
    patch = property(lambda self: self[2])