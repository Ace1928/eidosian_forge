import os, subprocess, json
class OldDeprecatedVersion:
    """
    A simple approach to Python package versioning that supports PyPI
    releases and additional information when working with version
    control. When obtaining a package from PyPI, the version returned
    is a string-formatted rendering of the supplied release tuple.
    For instance, release (1,0) tagged as ``v1.0`` in the version
    control system will return ``1.0`` for ``str(__version__)``.  Any
    number of items can be supplied in the release tuple, with either
    two or three numeric versioning levels typical.

    During development, a command like ``git describe`` will be used to
    compute the number of commits since the last version tag, the
    short commit hash, and whether the commit is dirty (has changes
    not yet committed). Version tags must start with a lowercase 'v'
    and have a period in them, e.g. v2.0, v0.9.8 or v0.1.

    Development versions are supported by setting the dev argument to an
    appropriate dev version number. The corresponding tag can be PEP440
    compliant (using .devX) of the form v0.1.dev3, v1.9.0.dev2 etc but
    it doesn't have to be as the dot may be omitted i.e v0.1dev3,
    v1.9.0dev2 etc.

    Also note that when version control system (VCS) information is
    used, the comparison operators take into account the number of
    commits since the last version tag. This approach is often useful
    in practice to decide which version is newer for a single
    developer, but will not necessarily be reliable when comparing
    against a different fork or branch in a distributed VCS.

    For git, if you want version control information available even in
    an exported archive (e.g. a .zip file from GitHub), you can set
    the following line in the .gitattributes file of your project::

      __init__.py export-subst
    """

    def __init__(self, release=None, fpath=None, commit=None, reponame=None, dev=None, commit_count=0):
        """
        :release:      Release tuple (corresponding to the current VCS tag)
        :commit        Short SHA. Set to '$Format:%h$' for git archive support.
        :fpath:        Set to ``__file__`` to access version control information
        :reponame:     Used to verify VCS repository name.
        :dev:          Development version number. None if not a development version.
        :commit_count  Commits since last release. Set for dev releases.
        """
        self.fpath = fpath
        self._expected_commit = commit
        self.expected_release = release
        self._commit = None if commit in [None, '$Format:%h$'] else commit
        self._commit_count = commit_count
        self._release = None
        self._dirty = False
        self.reponame = reponame
        self.dev = dev

    @property
    def release(self):
        """Return the release tuple"""
        return self.fetch()._release

    @property
    def commit(self):
        """A specification for this particular VCS version, e.g. a short git SHA"""
        return self.fetch()._commit

    @property
    def commit_count(self):
        """Return the number of commits since the last release"""
        return self.fetch()._commit_count

    @property
    def dirty(self):
        """True if there are uncommited changes, False otherwise"""
        return self.fetch()._dirty

    def fetch(self):
        """
        Returns a tuple of the major version together with the
        appropriate SHA and dirty bit (for development version only).
        """
        if self._release is not None:
            return self
        self._release = self.expected_release
        if not self.fpath:
            self._commit = self._expected_commit
            return self
        for cmd in ['git', 'git.cmd', 'git.exe']:
            try:
                self.git_fetch(cmd)
                break
            except OSError:
                pass
        return self

    def git_fetch(self, cmd='git'):
        try:
            if self.reponame is not None:
                output = run_cmd([cmd, 'remote', '-v'], cwd=os.path.dirname(self.fpath))
                repo_matches = ['/' + self.reponame + '.git', '/' + self.reponame + ' ']
                if not any((m in output for m in repo_matches)):
                    return self
            output = run_cmd([cmd, 'describe', '--long', '--match', 'v*.*', '--dirty'], cwd=os.path.dirname(self.fpath))
        except Exception as e:
            if e.args[1] == 'fatal: No names found, cannot describe anything.':
                raise Exception('Cannot find any git version tags of format v*.*')
            return self
        self._update_from_vcs(output)

    def _update_from_vcs(self, output):
        """Update state based on the VCS state e.g the output of git describe"""
        split = output[1:].split('-')
        if 'dev' in split[0]:
            dev_split = split[0].split('dev')
            self.dev = int(dev_split[1])
            split[0] = dev_split[0]
            if split[0].endswith('.'):
                split[0] = dev_split[0][:-1]
        self._release = tuple((int(el) for el in split[0].split('.')))
        self._commit_count = int(split[1])
        self._commit = str(split[2][1:])
        self._dirty = split[-1] == 'dirty'
        return self

    def __str__(self):
        """
        Version in x.y.z string format. Does not include the "v"
        prefix of the VCS version tags, for pip compatibility.

        If the commit count is non-zero or the repository is dirty,
        the string representation is equivalent to the output of::

          git describe --long --match v*.* --dirty

        (with "v" prefix removed).
        """
        if self.release is None:
            return 'None'
        release = '.'.join((str(el) for el in self.release))
        release = '%s.dev%d' % (release, self.dev) if self.dev is not None else release
        if self._expected_commit is not None and '$Format' not in self._expected_commit:
            pass
        elif self.commit_count == 0 and (not self.dirty):
            return release
        dirty_status = '-dirty' if self.dirty else ''
        return '{}-{}-g{}{}'.format(release, self.commit_count if self.commit_count else 'x', self.commit, dirty_status)

    def __repr__(self):
        return str(self)

    def abbrev(self, dev_suffix=''):
        """
        Abbreviated string representation, optionally declaring whether it is
        a development version.
        """
        return '.'.join((str(el) for el in self.release)) + (dev_suffix if self.commit_count > 0 or self.dirty else '')

    def __eq__(self, other):
        """
        Two versions are considered equivalent if and only if they are
        from the same release, with the same commit count, and are not
        dirty.  Any dirty version is considered different from any
        other version, since it could potentially have any arbitrary
        changes even for the same release and commit count.
        """
        if self.dirty or other.dirty:
            return False
        return (self.release, self.commit_count, self.dev) == (other.release, other.commit_count, other.dev)

    def __gt__(self, other):
        if self.release == other.release:
            if self.dev == other.dev:
                return self.commit_count > other.commit_count
            elif None in [self.dev, other.dev]:
                return self.dev is None
            else:
                return self.dev > other.dev
        else:
            return (self.release, self.commit_count) > (other.release, other.commit_count)

    def __lt__(self, other):
        if self == other:
            return False
        else:
            return not self > other

    def verify(self, string_version=None):
        """
        Check that the version information is consistent with the VCS
        before doing a release. If supplied with a string version,
        this is also checked against the current version. Should be
        called from setup.py with the declared package version before
        releasing to PyPI.
        """
        if string_version and string_version != str(self):
            raise Exception('Supplied string version does not match current version.')
        if self.dirty:
            raise Exception('Current working directory is dirty.')
        if self.release != self.expected_release:
            raise Exception('Declared release does not match current release tag.')
        if self.commit_count != 0:
            raise Exception('Please update the VCS version tag before release.')
        if self._expected_commit not in [None, '$Format:%h$']:
            raise Exception('Declared release does not match the VCS version tag')