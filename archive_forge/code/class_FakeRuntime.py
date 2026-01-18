class FakeRuntime(object):

    def get_version(self):
        return (-1, -1)

    def is_supported_version(self):
        return True

    @property
    def supported_versions(self):
        return ((-1, -1),)