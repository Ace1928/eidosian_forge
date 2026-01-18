from io import StringIO
from ... import branch as _mod_branch
from ... import controldir, errors
from ... import forge as _mod_forge
from ... import log as _mod_log
from ... import missing as _mod_missing
from ... import msgeditor, urlutils
from ...commands import Command
from ...i18n import gettext
from ...option import ListOption, Option, RegistryOption
from ...trace import note, warning
class cmd_propose_merge(Command):
    __doc__ = 'Propose a branch for merging.\n\n    This command creates a merge proposal for the local\n    branch to the target branch. The format of the merge\n    proposal depends on the submit branch.\n    '
    takes_options = ['directory', RegistryOption('forge', help='Use the forge.', lazy_registry=('breezy.forge', 'forges')), ListOption('reviewers', short_name='R', type=str, help='Requested reviewers.'), Option('name', help='Name of the new remote branch.', type=str), Option('description', help='Description of the change.', type=str), Option('prerequisite', help='Prerequisite branch.', type=str), Option('wip', help='Mark merge request as work-in-progress'), Option('auto', help='Automatically merge when the CI passes'), Option('commit-message', help='Set commit message for merge, if supported', type=str), ListOption('labels', short_name='l', type=str, help='Labels to apply.'), Option('no-allow-lossy', help='Allow fallback to lossy push, if necessary.'), Option('allow-collaboration', help='Allow collaboration from target branch maintainer(s)'), Option('allow-empty', help='Do not prevent empty merge proposals.'), Option('overwrite', help='Overwrite existing commits.'), Option('open', help='Open merge proposal in web browser'), Option('delete-source-after-merge', help='Delete source branch when proposal is merged'), 'revision']
    takes_args = ['submit_branch?']
    aliases = ['propose']

    def run(self, submit_branch=None, directory='.', forge=None, reviewers=None, name=None, no_allow_lossy=False, description=None, labels=None, prerequisite=None, commit_message=None, wip=False, allow_collaboration=False, allow_empty=False, overwrite=False, open=False, auto=False, delete_source_after_merge=None, revision=None):
        tree, branch, relpath = controldir.ControlDir.open_containing_tree_or_branch(directory)
        if submit_branch is None:
            submit_branch = branch.get_submit_branch()
        if submit_branch is None:
            submit_branch = branch.get_parent()
        if submit_branch is None:
            raise errors.CommandError(gettext('No target location specified or remembered'))
        target = _mod_branch.Branch.open(submit_branch)
        if not allow_empty:
            _check_already_merged(branch, target)
        if forge is None:
            forge = _mod_forge.get_forge(target)
        else:
            forge = forge.probe(target)
        if name is None:
            name = branch_name(branch)
        if revision is None:
            stop_revision = None
        else:
            stop_revision = revision.as_revision_id(branch)
        remote_branch, public_branch_url = forge.publish_derived(branch, target, name=name, allow_lossy=not no_allow_lossy, overwrite=overwrite, revision_id=stop_revision)
        branch.set_push_location(remote_branch.user_url)
        branch.set_submit_branch(target.user_url)
        note(gettext('Published branch to %s'), forge.get_web_url(remote_branch) or public_branch_url)
        if prerequisite is not None:
            prerequisite_branch = _mod_branch.Branch.open(prerequisite)
        else:
            prerequisite_branch = None
        proposal_builder = forge.get_proposer(remote_branch, target)
        if description is None:
            body = proposal_builder.get_initial_body()
            info = proposal_builder.get_infotext()
            info += '\n\n' + summarize_unmerged(branch, remote_branch, target, prerequisite_branch)
            description = msgeditor.edit_commit_message(info, start_message=body)
        try:
            proposal = proposal_builder.create_proposal(description=description, reviewers=reviewers, prerequisite_branch=prerequisite_branch, labels=labels, commit_message=commit_message, work_in_progress=wip, allow_collaboration=allow_collaboration, delete_source_after_merge=delete_source_after_merge)
        except _mod_forge.MergeProposalExists as e:
            note(gettext('There is already a branch merge proposal: %s'), e.url)
        else:
            note(gettext('Merge proposal created: %s') % proposal.url)
            if open:
                web_url = proposal.get_web_url()
                import webbrowser
                note(gettext('Opening %s in web browser'), web_url)
                webbrowser.open(web_url)
            if auto:
                proposal.merge(auto=True)