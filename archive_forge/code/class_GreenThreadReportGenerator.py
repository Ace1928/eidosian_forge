import gc
import sys
import threading
from oslo_reports.models import threading as tm
from oslo_reports.models import with_default_views as mwdv
from oslo_reports.views.text import generic as text_views
class GreenThreadReportGenerator(object):
    """A Green Thread Data Generator

    This generator returns a collection of
    :class:`oslo_reports.models.threading.GreenThreadModel`
    objects by introspecting the current python garbage collection
    state, and sifting through for :class:`greenlet.greenlet` objects.

    .. seealso::

        Function :func:`_find_objects`
    """

    def __call__(self):
        import greenlet
        threadModels = [tm.GreenThreadModel(gr.gr_frame) for gr in _find_objects(greenlet.greenlet)]
        return mwdv.ModelWithDefaultViews(threadModels, text_view=text_views.MultiView())