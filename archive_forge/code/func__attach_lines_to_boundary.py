from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
def _attach_lines_to_boundary(self, multi_line_strings, is_ccw):
    """
        Return a list of LinearRings by attaching the ends of the given lines
        to the boundary, paying attention to the traversal directions of the
        lines and boundary.

        """
    debug = False
    debug_plot_edges = False
    edge_things = []
    if is_ccw:
        boundary = self.ccw_boundary
    else:
        boundary = self.cw_boundary

    def boundary_distance(xy):
        return boundary.project(sgeom.Point(*xy))
    line_strings = []
    for multi_line_string in multi_line_strings:
        line_strings.extend(multi_line_string.geoms)
    for i, line_string in enumerate(line_strings):
        first_dist = boundary_distance(line_string.coords[0])
        thing = _BoundaryPoint(first_dist, False, (i, 'first', line_string.coords[0]))
        edge_things.append(thing)
        last_dist = boundary_distance(line_string.coords[-1])
        thing = _BoundaryPoint(last_dist, False, (i, 'last', line_string.coords[-1]))
        edge_things.append(thing)
    for xy in boundary.coords[:-1]:
        point = sgeom.Point(*xy)
        dist = boundary.project(point)
        thing = _BoundaryPoint(dist, True, point)
        edge_things.append(thing)
    if debug_plot_edges:
        import matplotlib.pyplot as plt
        current_fig = plt.gcf()
        fig = plt.figure()
        plt.figure(current_fig.number)
        ax = fig.add_subplot(1, 1, 1)
    edge_things.sort(key=lambda thing: (thing.distance, thing.kind))
    remaining_ls = dict(enumerate(line_strings))
    prev_thing = None
    for edge_thing in edge_things[:]:
        if prev_thing is not None and (not edge_thing.kind) and (not prev_thing.kind) and (edge_thing.data[0] == prev_thing.data[0]):
            j = edge_thing.data[0]
            mid_dist = (edge_thing.distance + prev_thing.distance) * 0.5
            mid_point = boundary.interpolate(mid_dist)
            new_thing = _BoundaryPoint(mid_dist, True, mid_point)
            if debug:
                print(f'Artificially insert boundary: {new_thing}')
            ind = edge_things.index(edge_thing)
            edge_things.insert(ind, new_thing)
            prev_thing = None
        else:
            prev_thing = edge_thing
    if debug:
        print()
        print('Edge things')
        for thing in edge_things:
            print('   ', thing)
    if debug_plot_edges:
        for thing in edge_things:
            if isinstance(thing.data, sgeom.Point):
                ax.plot(*thing.data.xy, marker='o')
            else:
                ax.plot(*thing.data[2], marker='o')
                ls = line_strings[thing.data[0]]
                coords = np.array(ls.coords)
                ax.plot(coords[:, 0], coords[:, 1])
                ax.text(coords[0, 0], coords[0, 1], thing.data[0])
                ax.text(coords[-1, 0], coords[-1, 1], f'{thing.data[0]}.')

    def filter_last(t):
        return t.kind or t.data[1] == 'first'
    edge_things = list(filter(filter_last, edge_things))
    processed_ls = []
    while remaining_ls:
        i, current_ls = remaining_ls.popitem()
        if debug:
            import sys
            sys.stdout.write('+')
            sys.stdout.flush()
            print()
            print(f'Processing: {i}, {current_ls}')
        added_linestring = set()
        while True:
            d_last = boundary_distance(current_ls.coords[-1])
            if debug:
                print(f'   d_last: {d_last!r}')
            next_thing = _find_first_ge(edge_things, d_last)
            edge_things.remove(next_thing)
            if debug:
                print('   next_thing:', next_thing)
            if next_thing.kind:
                if debug:
                    print('   adding boundary point')
                boundary_point = next_thing.data
                combined_coords = list(current_ls.coords) + [(boundary_point.x, boundary_point.y)]
                current_ls = sgeom.LineString(combined_coords)
            elif next_thing.data[0] == i:
                if debug:
                    print('   close loop')
                processed_ls.append(current_ls)
                if debug_plot_edges:
                    coords = np.array(current_ls.coords)
                    ax.plot(coords[:, 0], coords[:, 1], color='black', linestyle='--')
                break
            else:
                if debug:
                    print('   adding line')
                j = next_thing.data[0]
                line_to_append = line_strings[j]
                if j in remaining_ls:
                    remaining_ls.pop(j)
                coords_to_append = list(line_to_append.coords)
                current_ls = sgeom.LineString(list(current_ls.coords) + coords_to_append)
                if j not in added_linestring:
                    added_linestring.add(j)
                else:
                    if debug_plot_edges:
                        plt.show()
                    raise RuntimeError('Unidentified problem with geometry, linestring being re-added. Please raise an issue.')

    def makes_valid_ring(line_string):
        if len(line_string.coords) == 3:
            coords = list(line_string.coords)
            return coords[0] != coords[-1] and line_string.is_valid
        else:
            return len(line_string.coords) > 3 and line_string.is_valid
    linear_rings = [sgeom.LinearRing(line_string) for line_string in processed_ls if makes_valid_ring(line_string)]
    if debug:
        print('   DONE')
    return linear_rings