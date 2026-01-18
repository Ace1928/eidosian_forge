import os
class _Rule(object):

    class __metaclass__(type):

        def __new__(meta, class_name, bases, d):
            cls = type.__new__(meta, class_name, bases, d)
            PermissionSpec.commands[cls.__name__] = cls
            return cls
    inherit = False

    def noexists(self):
        return ['Path %s does not exist' % path]