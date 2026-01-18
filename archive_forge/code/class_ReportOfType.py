from oslo_reports.views.text import header as header_views
class ReportOfType(BasicReport):
    """A Report of a Certain Type

    A ReportOfType has a predefined type associated with it.
    This type is automatically propagated down to the each of
    the sections upon serialization by wrapping the generator
    for each section.

    .. seealso::

       Class :class:`oslo_reports.models.with_default_view.ModelWithDefaultView` # noqa
          (the entire class)

       Class :class:`oslo_reports.models.base.ReportModel`
           :func:`oslo_reports.models.base.ReportModel.set_current_view_type` # noqa

    :param str tp: the type of the report
    """

    def __init__(self, tp):
        self.output_type = tp
        super(ReportOfType, self).__init__()

    def add_section(self, view, generator, index=None):

        def with_type(gen):

            def newgen():
                res = gen()
                try:
                    res.set_current_view_type(self.output_type)
                except AttributeError:
                    pass
                return res
            return newgen
        super(ReportOfType, self).add_section(view, with_type(generator), index)