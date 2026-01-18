from typing import List, Optional, Union
from . import errors, hooks, osutils, trace, tree
class PostCommitHookParams:
    """Parameters for the post_commit hook.

    To access the parameters, use the following attributes:

    * mutable_tree - the MutableTree object
    """

    def __init__(self, mutable_tree):
        """Create the parameters for the post_commit hook."""
        self.mutable_tree = mutable_tree