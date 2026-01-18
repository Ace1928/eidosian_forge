import copy
import jinja2
def _gettext(self):
    """Get the Template Text

        Gets the text of the current template

        :returns: the text of the Jinja template
        :rtype: str
        """
    return self._text