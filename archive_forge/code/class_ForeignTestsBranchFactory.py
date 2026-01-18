import os
import dulwich
from dulwich.objects import Commit, Tag
from dulwich.repo import Repo as GitRepo
from ... import errors, revision, urlutils
from ...branch import Branch, InterBranch, UnstackableBranchFormat
from ...controldir import ControlDir
from ...repository import Repository
from .. import branch, tests
from ..dir import LocalGitControlDirFormat
from ..mapping import default_mapping
class ForeignTestsBranchFactory:

    def make_empty_branch(self, transport):
        d = LocalGitControlDirFormat().initialize_on_transport(transport)
        return d.create_branch()
    make_branch = make_empty_branch