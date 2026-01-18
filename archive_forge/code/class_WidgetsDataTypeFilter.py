import os
from warnings import warn
from nbconvert.utils.base import NbConvertBase
class WidgetsDataTypeFilter(NbConvertBase):
    """Returns the preferred display format, excluding the widget output if
    there is no widget state available"""

    def __init__(self, notebook_metadata=None, resources=None, **kwargs):
        """Initialize the filter."""
        self.metadata = notebook_metadata
        self.notebook_path = ''
        if resources is not None:
            name = resources.get('metadata', {}).get('name', '')
            path = resources.get('metadata', {}).get('path', '')
            self.notebook_path = os.path.join(path, name)
        super().__init__(**kwargs)

    def __call__(self, output):
        """Return the first available format in the priority.

        Produces a UserWarning if no compatible mimetype is found.

        `output` is dict with structure {mimetype-of-element: value-of-element}

        """
        metadata = self.metadata.get(self.notebook_path, {})
        widgets_state = metadata['widgets'][WIDGET_STATE_MIMETYPE]['state'] if metadata.get('widgets') is not None else {}
        for fmt in self.display_data_priority:
            if fmt in output:
                if fmt == WIDGET_VIEW_MIMETYPE and output[WIDGET_VIEW_MIMETYPE]['model_id'] not in widgets_state:
                    continue
                return [fmt]
        warn(f'Your element with mimetype(s) {output.keys()} is not able to be represented.', stacklevel=2)
        return []