import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
class TableFinder:
    """
    Given a PDF page, find plausible table structures.

    Largely borrowed from Anssi Nurminen's master's thesis:
    http://dspace.cc.tut.fi/dpub/bitstream/handle/123456789/21520/Nurminen.pdf?sequence=3

    ... and inspired by Tabula:
    https://github.com/tabulapdf/tabula-extractor/issues/16
    """

    def __init__(self, page, settings=None):
        self.page = page
        self.settings = TableSettings.resolve(settings)
        self.edges = self.get_edges()
        self.intersections = edges_to_intersections(self.edges, self.settings.intersection_x_tolerance, self.settings.intersection_y_tolerance)
        self.cells = intersections_to_cells(self.intersections)
        self.tables = [Table(self.page, cell_group) for cell_group in cells_to_tables(self.page, self.cells)]

    def get_edges(self) -> list:
        settings = self.settings
        for orientation in ['vertical', 'horizontal']:
            strategy = getattr(settings, orientation + '_strategy')
            if strategy == 'explicit':
                lines = getattr(settings, 'explicit_' + orientation + '_lines')
                if len(lines) < 2:
                    raise ValueError(f"If {orientation}_strategy == 'explicit', explicit_{orientation}_lines must be specified as a list/tuple of two or more floats/ints.")
        v_strat = settings.vertical_strategy
        h_strat = settings.horizontal_strategy
        if v_strat == 'text' or h_strat == 'text':
            words = extract_words(CHARS, **settings.text_settings or {})
        v_explicit = []
        for desc in settings.explicit_vertical_lines or []:
            if isinstance(desc, dict):
                for e in obj_to_edges(desc):
                    if e['orientation'] == 'v':
                        v_explicit.append(e)
            else:
                v_explicit.append({'x0': desc, 'x1': desc, 'top': self.page.rect[1], 'bottom': self.page.rect[3], 'height': self.page.rect[3] - self.page.rect[1], 'orientation': 'v'})
        if v_strat == 'lines':
            v_base = filter_edges(EDGES, 'v')
        elif v_strat == 'lines_strict':
            v_base = filter_edges(EDGES, 'v', edge_type='line')
        elif v_strat == 'text':
            v_base = words_to_edges_v(words, word_threshold=settings.min_words_vertical)
        elif v_strat == 'explicit':
            v_base = []
        v = v_base + v_explicit
        h_explicit = []
        for desc in settings.explicit_horizontal_lines or []:
            if isinstance(desc, dict):
                for e in obj_to_edges(desc):
                    if e['orientation'] == 'h':
                        h_explicit.append(e)
            else:
                h_explicit.append({'x0': self.page.rect[0], 'x1': self.page.rect[2], 'width': self.page.rect[2] - self.page.rect[0], 'top': desc, 'bottom': desc, 'orientation': 'h'})
        if h_strat == 'lines':
            h_base = filter_edges(EDGES, 'h')
        elif h_strat == 'lines_strict':
            h_base = filter_edges(EDGES, 'h', edge_type='line')
        elif h_strat == 'text':
            h_base = words_to_edges_h(words, word_threshold=settings.min_words_horizontal)
        elif h_strat == 'explicit':
            h_base = []
        h = h_base + h_explicit
        edges = list(v) + list(h)
        edges = merge_edges(edges, snap_x_tolerance=settings.snap_x_tolerance, snap_y_tolerance=settings.snap_y_tolerance, join_x_tolerance=settings.join_x_tolerance, join_y_tolerance=settings.join_y_tolerance)
        return filter_edges(edges, min_length=settings.edge_min_length)

    def __getitem__(self, i):
        tcount = len(self.tables)
        if i >= tcount:
            raise IndexError('table not on page')
        while i < 0:
            i += tcount
        return self.tables[i]