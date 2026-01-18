class VersionRequest:

    def __init__(self, version=None, min_api_version=None, max_api_version=None, default_microversion=None):
        self.version = version
        self.min_api_version = min_api_version
        self.max_api_version = max_api_version
        self.default_microversion = default_microversion