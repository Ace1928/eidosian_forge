import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
class Compositor(param.Parameterized):
    """
    A Compositor is a way of specifying an operation to be automatically
    applied to Overlays that match a specified pattern upon display.

    Any Operation that takes an Overlay as input may be used to define a
    compositor.

    For instance, a compositor may be defined to automatically display
    three overlaid monochrome matrices as an RGB image as long as the
    values names of those matrices match 'R', 'G' and 'B'.
    """
    mode = param.ObjectSelector(default='data', objects=['data', 'display'], doc="\n      The mode of the Compositor object which may be either 'data' or\n      'display'.")
    backends = param.List(default=[], doc='\n      Defines which backends to apply the Compositor for.')
    operation = param.Parameter(doc='\n       The Operation to apply when collapsing overlays.')
    pattern = param.String(doc="\n       The overlay pattern to be processed. An overlay pattern is a\n       sequence of elements specified by dotted paths separated by * .\n\n       For instance the following pattern specifies three overlaid\n       matrices with values of 'RedChannel', 'GreenChannel' and\n       'BlueChannel' respectively:\n\n      'Image.RedChannel * Image.GreenChannel * Image.BlueChannel.\n\n      This pattern specification could then be associated with the RGB\n      operation that returns a single RGB matrix for display.")
    group = param.String(allow_None=True, doc='\n       The group identifier for the output of this particular compositor')
    kwargs = param.Dict(doc='\n       Optional set of parameters to pass to the operation.')
    transfer_options = param.Boolean(default=False, doc='\n       Whether to transfer the options from the input to the output.')
    transfer_parameters = param.Boolean(default=False, doc='\n       Whether to transfer plot options which match to the operation.')
    operations = []
    definitions = []

    @classmethod
    def strongest_match(cls, overlay, mode, backend=None):
        """
        Returns the single strongest matching compositor operation
        given an overlay. If no matches are found, None is returned.

        The best match is defined as the compositor operation with the
        highest match value as returned by the match_level method.
        """
        match_strength = [(op.match_level(overlay), op) for op in cls.definitions if op.mode == mode and (not op.backends or backend in op.backends)]
        matches = [(match[0], op, match[1]) for match, op in match_strength if match is not None]
        if matches == []:
            return None
        return sorted(matches)[0]

    @classmethod
    def collapse_element(cls, overlay, ranges=None, mode='data', backend=None):
        """
        Finds any applicable compositor and applies it.
        """
        from .element import Element
        from .overlay import CompositeOverlay, Overlay
        unpack = False
        if not isinstance(overlay, CompositeOverlay):
            overlay = Overlay([overlay])
            unpack = True
        prev_ids = ()
        processed = defaultdict(list)
        while True:
            match = cls.strongest_match(overlay, mode, backend)
            if match is None:
                if unpack and len(overlay) == 1:
                    return overlay.values()[0]
                return overlay
            _, applicable_op, (start, stop) = match
            if isinstance(overlay, Overlay):
                values = overlay.values()
                sliced = Overlay(values[start:stop])
            else:
                values = overlay.items()
                sliced = overlay.clone(values[start:stop])
            items = sliced.traverse(lambda x: x, [Element])
            if applicable_op and all((el in processed[applicable_op] for el in items)):
                if unpack and len(overlay) == 1:
                    return overlay.values()[0]
                return overlay
            result = applicable_op.apply(sliced, ranges, backend)
            if applicable_op.group:
                result = result.relabel(group=applicable_op.group)
            if isinstance(overlay, Overlay):
                result = [result]
            else:
                result = list(zip(sliced.keys(), [result]))
            processed[applicable_op] += [el for r in result for el in r.traverse(lambda x: x, [Element])]
            overlay = overlay.clone(values[:start] + result + values[stop:])
            spec_fn = lambda x: not isinstance(x, CompositeOverlay)
            new_ids = tuple(overlay.traverse(lambda x: id(x), [spec_fn]))
            if new_ids == prev_ids:
                return overlay
            prev_ids = new_ids

    @classmethod
    def collapse(cls, holomap, ranges=None, mode='data'):
        """
        Given a map of Overlays, apply all applicable compositors.
        """
        if cls.definitions == []:
            return holomap
        clone = holomap.clone(shared_data=False)
        data = zip(ranges[1], holomap.data.values()) if ranges else holomap.data.items()
        for key, overlay in data:
            clone[key] = cls.collapse_element(overlay, ranges, mode)
        return clone

    @classmethod
    def map(cls, obj, mode='data', backend=None):
        """
        Applies compositor operations to any HoloViews element or container
        using the map method.
        """
        from .overlay import CompositeOverlay
        element_compositors = [c for c in cls.definitions if len(c._pattern_spec) == 1]
        overlay_compositors = [c for c in cls.definitions if len(c._pattern_spec) > 1]
        if overlay_compositors:
            obj = obj.map(lambda obj: cls.collapse_element(obj, mode=mode, backend=backend), [CompositeOverlay])
        element_patterns = [c.pattern for c in element_compositors]
        if element_compositors and obj.traverse(lambda x: x, element_patterns):
            obj = obj.map(lambda obj: cls.collapse_element(obj, mode=mode, backend=backend), element_patterns)
        return obj

    @classmethod
    def register(cls, compositor):
        defined_patterns = [op.pattern for op in cls.definitions]
        if compositor.pattern in defined_patterns:
            cls.definitions.pop(defined_patterns.index(compositor.pattern))
        cls.definitions.append(compositor)
        if compositor.operation not in cls.operations:
            cls.operations.append(compositor.operation)

    def __init__(self, pattern, operation, group, mode, transfer_options=False, transfer_parameters=False, output_type=None, backends=None, **kwargs):
        self._pattern_spec, labels = ([], [])
        for path in pattern.split('*'):
            path_tuple = tuple((el.strip() for el in path.strip().split('.')))
            self._pattern_spec.append(path_tuple)
            if len(path_tuple) == 3:
                labels.append(path_tuple[2])
        if len(labels) > 1 and (not all((l == labels[0] for l in labels))):
            raise KeyError('Mismatched labels not allowed in compositor patterns')
        elif len(labels) == 1:
            self.label = labels[0]
        else:
            self.label = ''
        self._output_type = output_type
        super().__init__(group=group, pattern=pattern, operation=operation, mode=mode, backends=backends or [], kwargs=kwargs, transfer_options=transfer_options, transfer_parameters=transfer_parameters)

    @property
    def output_type(self):
        """
        Returns the operation output_type unless explicitly overridden
        in the kwargs.
        """
        return self._output_type or self.operation.output_type

    def _slice_match_level(self, overlay_items):
        """
        Find the match strength for a list of overlay items that must
        be exactly the same length as the pattern specification.
        """
        level = 0
        for spec, el in zip(self._pattern_spec, overlay_items):
            if spec[0] != type(el).__name__:
                return None
            level += 1
            if len(spec) == 1:
                continue
            group = [el.group, group_sanitizer(el.group, escape=False)]
            if spec[1] in group:
                level += 1
            else:
                return None
            if len(spec) == 3:
                group = [el.label, label_sanitizer(el.label, escape=False)]
                if spec[2] in group:
                    level += 1
                else:
                    return None
        return level

    def match_level(self, overlay):
        """
        Given an overlay, return the match level and applicable slice
        of the overall overlay. The level an integer if there is a
        match or None if there is no match.

        The level integer is the number of matching components. Higher
        values indicate a stronger match.
        """
        slice_width = len(self._pattern_spec)
        if slice_width > len(overlay):
            return None
        best_lvl, match_slice = (0, None)
        for i in range(len(overlay) - slice_width + 1):
            overlay_slice = overlay.values()[i:i + slice_width]
            lvl = self._slice_match_level(overlay_slice)
            if lvl is None:
                continue
            if lvl > best_lvl:
                best_lvl = lvl
                match_slice = (i, i + slice_width)
        return (best_lvl, match_slice) if best_lvl != 0 else None

    def apply(self, value, input_ranges, backend=None):
        """
        Apply the compositor on the input with the given input ranges.
        """
        from .overlay import CompositeOverlay
        if backend is None:
            backend = Store.current_backend
        kwargs = {k: v for k, v in self.kwargs.items() if k != 'output_type'}
        if isinstance(value, CompositeOverlay) and len(value) == 1:
            value = value.values()[0]
            if self.transfer_parameters:
                plot_opts = Store.lookup_options(backend, value, 'plot').kwargs
                kwargs.update({k: v for k, v in plot_opts.items() if k in self.operation.param})
        transformed = self.operation(value, input_ranges=input_ranges, **kwargs)
        if self.transfer_options and value is not transformed:
            Store.transfer_options(value, transformed, backend)
        return transformed