import oslo_reports.models.with_default_views as mwdv
import oslo_reports.views.text.process as text_views
class ProcessModel(mwdv.ModelWithDefaultViews):
    """A Process Model

    This model holds data about a process,
    including references to any subprocesses

    :param process: a :class:`psutil.Process` object
    """

    def __init__(self, process):
        super(ProcessModel, self).__init__(text_view=text_views.ProcessView())
        self['pid'] = process.pid
        self['parent_pid'] = process.ppid()
        if hasattr(process, 'uids'):
            self['uids'] = {'real': process.uids().real, 'effective': process.uids().effective, 'saved': process.uids().saved}
        else:
            self['uids'] = {'real': None, 'effective': None, 'saved': None}
        if hasattr(process, 'gids'):
            self['gids'] = {'real': process.gids().real, 'effective': process.gids().effective, 'saved': process.gids().saved}
        else:
            self['gids'] = {'real': None, 'effective': None, 'saved': None}
        self['username'] = process.username()
        self['command'] = process.cmdline()
        self['state'] = process.status()
        children = process.children()
        self['children'] = [ProcessModel(pr) for pr in children]