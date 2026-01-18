from .. import config
class GitBranchStack(config._CompatibleStack):
    """GitBranch stack."""

    def __init__(self, branch):
        section_getters = [self._get_overrides]
        lstore = config.LocationStore()
        loc_matcher = config.LocationMatcher(lstore, branch.base)
        section_getters.append(loc_matcher.get_sections)
        git = getattr(branch.repository, '_git', None)
        if git:
            cstore = GitConfigStore('branch', git.get_config())
            section_getters.append(cstore.get_sections)
        gstore = config.GlobalStore()
        section_getters.append(gstore.get_sections)
        super().__init__(section_getters, lstore, branch.base)
        self.branch = branch