from copy import deepcopy
from warnings import warn
from traitlets import Bool, Unicode, default
from nbconvert.preprocessors.base import Preprocessor
from .html import HTMLExporter
class _RevealMetadataPreprocessor(Preprocessor):

    def preprocess(self, nb, resources=None):
        nb = deepcopy(nb)
        for cell in nb.cells:
            try:
                slide_type = cell.metadata.get('slideshow', {}).get('slide_type', '-')
            except AttributeError:
                slide_type = '-'
            cell.metadata.slide_type = slide_type
        for index, cell in enumerate(nb.cells):
            if cell.metadata.slide_type not in {'notes', 'skip'}:
                cell.metadata.slide_type = 'slide'
                cell.metadata.slide_start = True
                cell.metadata.subslide_start = True
                first_slide_ix = index
                break
        else:
            msg = 'All cells are hidden, cannot create slideshow'
            raise ValueError(msg)
        in_fragment = False
        for index, cell in enumerate(nb.cells[first_slide_ix + 1:], start=first_slide_ix + 1):
            previous_cell = nb.cells[index - 1]
            if cell.metadata.slide_type == 'slide':
                previous_cell.metadata.slide_end = True
                cell.metadata.slide_start = True
            if cell.metadata.slide_type in {'subslide', 'slide'}:
                previous_cell.metadata.fragment_end = in_fragment
                previous_cell.metadata.subslide_end = True
                cell.metadata.subslide_start = True
                in_fragment = False
            elif cell.metadata.slide_type == 'fragment':
                cell.metadata.fragment_start = True
                if in_fragment:
                    previous_cell.metadata.fragment_end = True
                else:
                    in_fragment = True
        nb.cells[-1].metadata.fragment_end = in_fragment
        nb.cells[-1].metadata.subslide_end = True
        nb.cells[-1].metadata.slide_end = True
        return (nb, resources)