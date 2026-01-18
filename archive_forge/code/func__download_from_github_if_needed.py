import hashlib
import importlib.util
import os
import re
import subprocess
import tempfile
import yaml
import ray
def _download_from_github_if_needed(config_path: str) -> str:
    """Resolve a GitHub raw link to the config file to a local path.

    If the user specifies a GitHub raw URL, download the repo specified at
    that particular URL locally. This lets us treat YAMLs linked from GitHub
    the same as local files.
    """
    if config_path.startswith('http'):
        if 'github' not in config_path:
            raise ValueError('Only GitHub URLs are supported by load_package().')
        if 'raw.githubusercontent.com' not in config_path:
            raise ValueError('GitHub URL must start with raw.githubusercontent.com')
        URL_FORMAT = '.*raw.githubusercontent.com/([^/]*)/([^/]*)/([^/]*)/(.*)'
        match = re.match(URL_FORMAT, config_path)
        if not match:
            raise ValueError('GitHub URL must be of format {}'.format(URL_FORMAT))
        gh_user = match.group(1)
        gh_repo = match.group(2)
        gh_branch = match.group(3)
        gh_subdir = match.group(4)
        hasher = hashlib.sha1()
        hasher.update(config_path.encode('utf-8'))
        config_key = hasher.hexdigest()
        final_path = os.path.join(_pkg_tmp(), 'github_snapshot_{}'.format(config_key))
        if not os.path.exists(final_path):
            tmp = tempfile.mkdtemp(prefix='github_{}'.format(gh_repo), dir=_pkg_tmp())
            subprocess.check_call(['curl', '--fail', '-L', 'https://github.com/{}/{}/tarball/{}'.format(gh_user, gh_repo, gh_branch), '--output', tmp + '.tar.gz'])
            subprocess.check_call(['tar', 'xzf', tmp + '.tar.gz', '-C', tmp, '--strip-components=1'])
            os.rename(tmp, final_path)
        return os.path.join(final_path, gh_subdir)
    return config_path