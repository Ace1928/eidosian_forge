import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
class Story:

    def __init__(self, html='', user_css=None, em=12, archive=None):
        buffer_ = mupdf.fz_new_buffer_from_copied_data(html.encode('utf-8'))
        if archive and (not isinstance(archive, Archive)):
            archive = Archive(archive)
        arch = archive.this if archive else mupdf.FzArchive(None)
        if hasattr(mupdf, 'FzStoryS'):
            self.this = mupdf.FzStoryS(buffer_, user_css, em, arch)
        else:
            self.this = mupdf.FzStory(buffer_, user_css, em, arch)

    def add_header_ids(self):
        """
        Look for `<h1..6>` items in `self` and adds unique `id`
        attributes if not already present.
        """
        dom = self.body
        i = 0
        x = dom.find(None, None, None)
        while x:
            name = x.tagname
            if len(name) == 2 and name[0] == 'h' and (name[1] in '123456'):
                attr = x.get_attribute_value('id')
                if not attr:
                    id_ = f'h_id_{i}'
                    x.set_attribute('id', id_)
                    i += 1
            x = x.find_next(None, None, None)

    @staticmethod
    def add_pdf_links(document_or_stream, positions):
        """
        Adds links to PDF document.
        Args:
            document_or_stream:
                A PDF `Document` or raw PDF content, for example an
                `io.BytesIO` instance.
            positions:
                List of `ElementPosition`'s for `document_or_stream`,
                typically from Story.element_positions(). We raise an
                exception if two or more positions have same id.
        Returns:
            `document_or_stream` if a `Document` instance, otherwise a
            new `Document` instance.
        We raise an exception if an `href` in `positions` refers to an
        internal position `#<name>` but no item in `postions` has `id =
        name`.
        """
        if isinstance(document_or_stream, Document):
            document = document_or_stream
        else:
            document = Document('pdf', document_or_stream)
        id_to_position = dict()
        for position in positions:
            if position.open_close & 1 and position.id:
                if position.id in id_to_position:
                    pass
                else:
                    id_to_position[position.id] = position
        for position_from in positions:
            if position_from.open_close & 1 and position_from.href:
                link = dict()
                link['from'] = Rect(position_from.rect)
                if position_from.href.startswith('#'):
                    target_id = position_from.href[1:]
                    try:
                        position_to = id_to_position[target_id]
                    except Exception as e:
                        if g_exceptions_verbose > 1:
                            exception_info()
                        raise RuntimeError(f'No destination with id={target_id}, required by position_from: {position_from}') from e
                    if 0:
                        log(f'add_pdf_links(): making link from:')
                        log(f'add_pdf_links():    {position_from}')
                        log(f'add_pdf_links(): to:')
                        log(f'add_pdf_links():    {position_to}')
                    link['kind'] = LINK_GOTO
                    x0, y0, x1, y1 = position_to.rect
                    link['to'] = Point(x0, y0)
                    link['page'] = position_to.page_num - 1
                elif position_from.href.startswith('name:'):
                    link['kind'] = LINK_NAMED
                    link['name'] = position_from.href[5:]
                else:
                    link['kind'] = LINK_URI
                    link['uri'] = position_from.href
                document[position_from.page_num - 1].insert_link(link)
        return document

    @property
    def body(self):
        dom = self.document()
        return dom.bodytag()

    def document(self):
        dom = mupdf.fz_story_document(self.this)
        return Xml(dom)

    def draw(self, device, matrix=None):
        ctm2 = JM_matrix_from_py(matrix)
        dev = device.this if device else mupdf.FzDevice(None)
        sys.stdout.flush()
        mupdf.fz_draw_story(self.this, dev, ctm2)

    def element_positions(self, function, args=None):
        """
        Trigger a callback function to record where items have been placed.
        """
        if type(args) is dict:
            for k in args.keys():
                if not (type(k) is str and k.isidentifier()):
                    raise ValueError(f"invalid key '{k}'")
        else:
            args = {}
        if not callable(function) or function.__code__.co_argcount != 1:
            raise ValueError("callback 'function' must be a callable with exactly one argument")

        def function2(position):

            class Position2:
                pass
            position2 = Position2()
            position2.depth = position.depth
            position2.heading = position.heading
            position2.id = position.id
            position2.rect = JM_py_from_rect(position.rect)
            position2.text = position.text
            position2.open_close = position.open_close
            position2.rect_num = position.rectangle_num
            position2.href = position.href
            if args:
                for k, v in args.items():
                    setattr(position2, k, v)
            function(position2)
        mupdf.fz_story_positions(self.this, function2)

    def place(self, where):
        where = JM_rect_from_py(where)
        filled = mupdf.FzRect()
        more = mupdf.fz_place_story(self.this, where, filled)
        return (more, JM_py_from_rect(filled))

    def reset(self):
        mupdf.fz_reset_story(self.this)

    def write(self, writer, rectfn, positionfn=None, pagefn=None):
        dev = None
        page_num = 0
        rect_num = 0
        filled = Rect(0, 0, 0, 0)
        while 1:
            mediabox, rect, ctm = rectfn(rect_num, filled)
            rect_num += 1
            if mediabox:
                page_num += 1
            more, filled = self.place(rect)
            if positionfn:

                def positionfn2(position):
                    position.page_num = page_num
                    positionfn(position)
                self.element_positions(positionfn2)
            if writer:
                if mediabox:
                    if dev:
                        if pagefn:
                            pagefn(page_num, mediabox, dev, 1)
                        writer.end_page()
                    dev = writer.begin_page(mediabox)
                    if pagefn:
                        pagefn(page_num, mediabox, dev, 0)
                self.draw(dev, ctm)
                if not more:
                    if pagefn:
                        pagefn(page_num, mediabox, dev, 1)
                    writer.end_page()
            else:
                self.draw(None, ctm)
            if not more:
                break

    @staticmethod
    def write_stabilized(writer, contentfn, rectfn, user_css=None, em=12, positionfn=None, pagefn=None, archive=None, add_header_ids=True):
        positions = list()
        content = None
        while 1:
            content_prev = content
            content = contentfn(positions)
            stable = False
            if content == content_prev:
                stable = True
            content2 = content
            story = Story(content2, user_css, em, archive)
            if add_header_ids:
                story.add_header_ids()
            positions = list()

            def positionfn2(position):
                positions.append(position)
                if stable and positionfn:
                    positionfn(position)
            story.write(writer if stable else None, rectfn, positionfn2, pagefn)
            if stable:
                break

    @staticmethod
    def write_stabilized_with_links(contentfn, rectfn, user_css=None, em=12, positionfn=None, pagefn=None, archive=None, add_header_ids=True):
        stream = io.BytesIO()
        writer = DocumentWriter(stream)
        positions = []

        def positionfn2(position):
            positions.append(position)
            if positionfn:
                positionfn(position)
        Story.write_stabilized(writer, contentfn, rectfn, user_css, em, positionfn2, pagefn, archive, add_header_ids)
        writer.close()
        stream.seek(0)
        return Story.add_pdf_links(stream, positions)

    def write_with_links(self, rectfn, positionfn=None, pagefn=None):
        stream = io.BytesIO()
        writer = DocumentWriter(stream)
        positions = []

        def positionfn2(position):
            positions.append(position)
            if positionfn:
                positionfn(position)
        self.write(writer, rectfn, positionfn=positionfn2, pagefn=pagefn)
        writer.close()
        stream.seek(0)
        return Story.add_pdf_links(stream, positions)

    class FitResult:
        """
        The result from a `Story.fit*()` method.
        
        Members:
        
        `big_enough`:
            `True` if the fit succeeded.
        `filled`:
            From the last call to `Story.place()`.
        `more`:
            `False` if the fit succeeded.
        `numcalls`:
            Number of calls made to `self.place()`.
        `parameter`:
            The successful parameter value, or the largest failing value.
        `rect`:
            The rect created from `parameter`.
        """

        def __init__(self, big_enough=None, filled=None, more=None, numcalls=None, parameter=None, rect=None):
            self.big_enough = big_enough
            self.filled = filled
            self.more = more
            self.numcalls = numcalls
            self.parameter = parameter
            self.rect = rect

        def __repr__(self):
            return f' big_enough={self.big_enough} filled={self.filled} more={self.more} numcalls={self.numcalls} parameter={self.parameter} rect={self.rect}'

    def fit(self, fn, pmin=None, pmax=None, delta=0.001, verbose=False):
        """
        Finds optimal rect that contains the story `self`.
        
        Returns a `Story.FitResult` instance.
            
        On success, the last call to `self.place()` will have been with the
        returned rectangle, so `self.draw()` can be used directly.
        
        Args:
        :arg fn:
            A callable taking a floating point `parameter` and returning a
            `fitz.Rect()`. If the rect is empty, we assume the story will
            not fit and do not call `self.place()`.

            Must guarantee that `self.place()` behaves monotonically when
            given rect `fn(parameter`) as `parameter` increases. This
            usually means that both width and height increase or stay
            unchanged as `parameter` increases.
        :arg pmin:
            Minimum parameter to consider; `None` for -infinity.
        :arg pmax:
            Maximum parameter to consider; `None` for +infinity.
        :arg delta:
            Maximum error in returned `parameter`.
        :arg verbose:
            If true we output diagnostics.
        """

        def log(text):
            assert verbose
            message(f'fit(): {text}')
            sys.stdout.flush()
        assert isinstance(pmin, (int, float)) or pmin is None
        assert isinstance(pmax, (int, float)) or pmax is None

        class State:

            def __init__(self):
                self.pmin = pmin
                self.pmax = pmax
                self.pmin_result = None
                self.pmax_result = None
                self.result = None
                self.numcalls = 0
                if verbose:
                    self.pmin0 = pmin
                    self.pmax0 = pmax
        state = State()
        if verbose:
            log(f'starting. state.pmin={state.pmin!r} state.pmax={state.pmax!r}.')
        self.reset()

        def ret():
            if state.pmax is not None:
                if state.last_p != state.pmax:
                    if verbose:
                        log(f'Calling update() with pmax, because was overwritten by later calls.')
                    big_enough = update(state.pmax)
                    assert big_enough
                result = state.pmax_result
            else:
                result = state.pmin_result if state.pmin_result else Story.FitResult(numcalls=state.numcalls)
            if verbose:
                log(f'finished. state.pmin0={state.pmin0!r} state.pmax0={state.pmax0!r} state.pmax={state.pmax!r}: returning result={result!r}')
            return result

        def update(parameter):
            """
            Evaluates `more, _ = self.place(fn(parameter))`. If `more` is
            false, then `rect` is big enought to contain `self` and we
            set `state.pmax=parameter` and return True. Otherwise we set
            `state.pmin=parameter` and return False.
            """
            rect = fn(parameter)
            assert isinstance(rect, Rect), f'type(rect)={type(rect)!r} rect={rect!r}'
            if rect.is_empty:
                big_enough = False
                result = Story.FitResult(parameter=parameter, numcalls=state.numcalls)
                if verbose:
                    log(f'update(): not calling self.place() because rect is empty.')
            else:
                more, filled = self.place(rect)
                state.numcalls += 1
                big_enough = not more
                result = Story.FitResult(filled=filled, more=more, numcalls=state.numcalls, parameter=parameter, rect=rect, big_enough=big_enough)
                if verbose:
                    log(f'update(): called self.place(): {state.numcalls:>2d}: more={more!r} parameter={parameter!r} rect={rect!r}.')
            if big_enough:
                state.pmax = parameter
                state.pmax_result = result
            else:
                state.pmin = parameter
                state.pmin_result = result
            state.last_p = parameter
            return big_enough

        def opposite(p, direction):
            """
            Returns same sign as `direction`, larger or smaller than `p` if
            direction is positive or negative respectively.
            """
            if p is None or p == 0:
                return direction
            if direction * p > 0:
                return 2 * p
            return -p
        if state.pmin is None:
            if verbose:
                log(f'finding pmin.')
            parameter = opposite(state.pmax, -1)
            while 1:
                if not update(parameter):
                    break
                parameter *= 2
        elif update(state.pmin):
            if verbose:
                log(f'state.pmin={state.pmin!r} is big enough.')
            return ret()
        if state.pmax is None:
            if verbose:
                log(f'finding pmax.')
            parameter = opposite(state.pmin, +1)
            while 1:
                if update(parameter):
                    break
                parameter *= 2
        elif not update(state.pmax):
            state.pmax = None
            if verbose:
                log(f'No solution possible state.pmax={state.pmax!r}.')
            return ret()
        if verbose:
            log(f'doing binary search with state.pmin={state.pmin!r} state.pmax={state.pmax!r}.')
        while 1:
            if state.pmax - state.pmin < delta:
                return ret()
            parameter = (state.pmin + state.pmax) / 2
            update(parameter)

    def fit_scale(self, rect, scale_min=0, scale_max=None, delta=0.001, verbose=False):
        """
        Finds smallest value `scale` in range `scale_min..scale_max` where
        `scale * rect` is large enough to contain the story `self`.

        Returns a `Story.FitResult` instance.

        :arg width:
            width of rect.
        :arg height:
            height of rect.
        :arg scale_min:
            Minimum scale to consider; must be >= 0.
        :arg scale_max:
            Maximum scale to consider, must be >= scale_min or `None` for
            infinite.
        :arg delta:
            Maximum error in returned scale.
        :arg verbose:
            If true we output diagnostics.
        """
        x0, y0, x1, y1 = rect
        width = x1 - x0
        height = y1 - y0

        def fn(scale):
            return Rect(x0, y0, x0 + scale * width, y0 + scale * height)
        return self.fit(fn, scale_min, scale_max, delta, verbose)

    def fit_height(self, width, height_min=0, height_max=None, origin=(0, 0), delta=0.001, verbose=False):
        """
        Finds smallest height in range `height_min..height_max` where a rect
        with size `(width, height)` is large enough to contain the story
        `self`.

        Returns a `Story.FitResult` instance.

        :arg width:
            width of rect.
        :arg height_min:
            Minimum height to consider; must be >= 0.
        :arg height_max:
            Maximum height to consider, must be >= height_min or `None` for
            infinite.
        :arg origin:
            `(x0, y0)` of rect.
        :arg delta:
            Maximum error in returned height.
        :arg verbose:
            If true we output diagnostics.
        """
        x0, y0 = origin
        x1 = x0 + width

        def fn(height):
            return Rect(x0, y0, x1, y0 + height)
        return self.fit(fn, height_min, height_max, delta, verbose)

    def fit_width(self, height, width_min=0, width_max=None, origin=(0, 0), delta=0.001, verbose=False):
        """
        Finds smallest width in range `width_min..width_max` where a rect with size
        `(width, height)` is large enough to contain the story `self`.

        Returns a `Story.FitResult` instance.
        Returns a `FitResult` instance.

        :arg height:
            height of rect.
        :arg width_min:
            Minimum width to consider; must be >= 0.
        :arg width_max:
            Maximum width to consider, must be >= width_min or `None` for
            infinite.
        :arg origin:
            `(x0, y0)` of rect.
        :arg delta:
            Maximum error in returned width.
        :arg verbose:
            If true we output diagnostics.
        """
        x0, y0 = origin
        y1 = x0 + height

        def fn(width):
            return Rect(x0, y0, x0 + width, y1)
        return self.fit(fn, width_min, width_max, delta, verbose)