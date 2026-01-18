class PermLookupDict:

    def __init__(self, user, app_label):
        self.user, self.app_label = (user, app_label)

    def __repr__(self):
        return str(self.user.get_all_permissions())

    def __getitem__(self, perm_name):
        return self.user.has_perm('%s.%s' % (self.app_label, perm_name))

    def __iter__(self):
        raise TypeError('PermLookupDict is not iterable.')

    def __bool__(self):
        return self.user.has_module_perms(self.app_label)