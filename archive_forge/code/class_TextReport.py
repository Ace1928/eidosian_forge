from oslo_reports.views.text import header as header_views
class TextReport(ReportOfType):
    """A Human-Readable Text Report

    This class defines a report that is designed to be read by a human
    being.  It has nice section headers, and a formatted title.

    :param str name: the title of the report
    """

    def __init__(self, name):
        super(TextReport, self).__init__('text')
        self.name = name
        self.add_section(name, lambda: '|' * 72 + '\n\n')

    def add_section(self, heading, generator, index=None):
        """Add a section to the report

        This method adds a section with the given title, and
        generator to the report.  An index may be specified to
        insert the section at a given location in the list;
        If no index is specified, the section is appended to the
        list.  The view is called on the model which results from
        the generator when the report is run.  A generator is simply
        a method or callable object which takes no arguments and
        returns a :class:`oslo_reports.models.base.ReportModel`
        or similar object.

        The model is told to serialize as text (if possible) at serialization
        time by wrapping the generator.  The view model's attached view
        (if any) is wrapped in a
        :class:`oslo_reports.views.text.header.TitledView`

        :param str heading: the title for the section
        :param generator: the method or class which generates the model
        :param index: the index at which to insert the section
                      (or None to append)
        :type index: int or None
        """
        super(TextReport, self).add_section(header_views.TitledView(heading), generator, index)