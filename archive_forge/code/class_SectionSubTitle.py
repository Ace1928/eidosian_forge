import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class SectionSubTitle(TitlePromoter):
    """
    This works like document subtitles, but for sections.  For example, ::

        <section>
            <title>
                Title
            <section>
                <title>
                    Subtitle
                ...

    is transformed into ::

        <section>
            <title>
                Title
            <subtitle>
                Subtitle
            ...

    For details refer to the docstring of DocTitle.
    """
    default_priority = 350

    def apply(self):
        if not getattr(self.document.settings, 'sectsubtitle_xform', 1):
            return
        for section in self.document.traverse(nodes.section):
            self.promote_subtitle(section)