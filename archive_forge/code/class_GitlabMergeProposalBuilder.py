import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, urlutils
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...trace import mutter
from ...transport import get_transport
class GitlabMergeProposalBuilder(MergeProposalBuilder):

    def __init__(self, gl, source_branch, target_branch):
        self.gl = gl
        self.source_branch = source_branch
        self.source_host, self.source_project_name, self.source_branch_name = parse_gitlab_branch_url(source_branch)
        self.target_branch = target_branch
        self.target_host, self.target_project_name, self.target_branch_name = parse_gitlab_branch_url(target_branch)
        if self.source_host != self.target_host:
            raise DifferentGitLabInstances(self.source_host, self.target_host)

    def get_infotext(self):
        """Determine the initial comment for the merge proposal."""
        info = []
        info.append('Gitlab instance: %s\n' % self.target_host)
        info.append('Source: %s\n' % self.source_branch.user_url)
        info.append('Target: %s\n' % self.target_branch.user_url)
        return ''.join(info)

    def get_initial_body(self):
        """Get a body for the proposal for the user to modify.

        :return: a str or None.
        """
        return None

    def create_proposal(self, description, title=None, reviewers=None, labels=None, prerequisite_branch=None, commit_message=None, work_in_progress=False, allow_collaboration=False, delete_source_after_merge: Optional[bool]=None):
        """Perform the submission."""
        if prerequisite_branch is not None:
            raise PrerequisiteBranchUnsupported(self)
        source_project = self.gl._get_project(self.source_project_name)
        target_project = self.gl._get_project(self.target_project_name)
        if title is None:
            title = determine_title(description)
        if work_in_progress:
            title = 'WIP: %s' % title
        kwargs = {'title': title, 'source_project_id': source_project['id'], 'target_project_id': target_project['id'], 'source_branch_name': self.source_branch_name, 'target_branch_name': self.target_branch_name, 'description': description, 'allow_collaboration': allow_collaboration}
        if delete_source_after_merge is not None:
            kwargs['should_remove_source_branch'] = delete_source_after_merge
        if labels:
            kwargs['labels'] = ','.join(labels)
        if reviewers:
            kwargs['assignee_ids'] = []
            for reviewer in reviewers:
                if '@' in reviewer:
                    user = self.gl._get_user_by_email(reviewer)
                else:
                    user = self.gl._get_user(reviewer)
                kwargs['assignee_ids'].append(user['id'])
        try:
            merge_request = self.gl._create_mergerequest(**kwargs)
        except GitLabConflict as e:
            self.gl._handle_merge_request_conflict(e.reason, self.source_branch.user_url, target_project['path_with_namespace'])
        except GitLabUnprocessable as e:
            if e.error == ['Source project is not a fork of the target project']:
                raise SourceNotDerivedFromTarget(self.source_branch, self.target_branch)
            raise
        return GitLabMergeProposal(self.gl, merge_request)