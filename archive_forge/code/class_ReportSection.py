from oslo_reports.views.text import header as header_views
class ReportSection(object):
    """A Report Section

    A report section contains a generator and a top-level view. When something
    attempts to serialize the section by calling str() or unicode() on it, the
    section runs the generator and calls the view on the resulting model.

    .. seealso::

       Class :class:`BasicReport`
           :func:`BasicReport.add_section`

    :param view: the top-level view for this section
    :param generator: the generator for this section
      (any callable object which takes no parameters and returns a data model)
    """

    def __init__(self, view, generator):
        self.view = view
        self.generator = generator

    def __str__(self):
        return self.view(self.generator())