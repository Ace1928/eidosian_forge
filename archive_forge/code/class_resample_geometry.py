import param
import numpy as np
from holoviews import Polygons, Path
from holoviews.streams import RangeXY
from holoviews import Operation
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from ..util import polygons_to_geom_dicts, path_to_geom_dicts, shapely_v2
class resample_geometry(Operation):
    """
    This operation dynamically culls and resamples Path or Polygons
    elements based on the current zoom level. On first execution a
    RTree is created using the Sort-Tile-Recursive algorithm, which is
    used to query for geometries within the current viewport (defined
    by the x_range and y_range).

    Any geometries returned by the RTree query are tested to ensure
    their area is over the display_threshold, expressed as a fraction
    of the current viewport area. Any remaining polygons are
    simplified using the Douglas-Peucker algorithm, which eliminates
    vertices while ensuring that the curve does not diverge from the
    original curve by more than the tolerance. The tolerance is
    expressed as a fraction of the square root of the area of the
    current viewport.

    Once computed a simplified geometry is cached depending on the
    current zoom level. The number of valid zoom levels can be
    declared and are used to recursively subdivide the domain into
    smaller subregions.

    If requested the geometries can also be clipped to the current
    viewport which avoids having to render vertices that are not
    visible.
    """
    cache = param.Boolean(default=True, doc='\n        Whether to cache simplified geometries depending on the zoom\n        level.')
    clip = param.Boolean(default=False, doc='\n        Whether to disable the cache and clip polygons\n        to current bounds.')
    display_threshold = param.Number(default=0.0001, doc='\n        The fraction of the current viewport covered by a geometry\n        before it is shown.')
    dynamic = param.Boolean(default=True, doc='\n       Enables dynamic processing by default.')
    preserve_topology = param.Boolean(default=False, doc='\n        Whether to preserve topology between geometries. If disabled\n        simplification can produce self-intersecting or otherwise\n        invalid geometries but will be much faster.')
    streams = param.ClassSelector(default=[RangeXY], class_=(dict, list), doc='\n        List or dictionary streams that are applied if dynamic=True,\n        allowing for dynamic interaction with the plot.')
    tolerance_factor = param.Number(default=0.002, doc='\n        The tolerance distance for path simplification as a fraction\n        of the square root of the area of the current viewport.')
    x_range = param.NumericTuple(default=None, length=2, doc='\n       The x_range as a tuple of min and max x-value. Auto-ranges\n       if set to None.')
    y_range = param.NumericTuple(default=None, length=2, doc='\n       The x_range as a tuple of min and max y-value. Auto-ranges\n       if set to None.')
    zoom_levels = param.Integer(default=20, doc='\n        The number of zoom levels to cache.')
    _per_element = True

    @param.parameterized.bothmethod
    def instance(self_or_cls, **params):
        inst = super().instance(**params)
        inst._cache = {}
        return inst

    def _process(self, element, key=None):
        x0, x1 = self.p.x_range or element.range(0)
        y0, y1 = self.p.y_range or element.range(1)
        bounds = bounds_to_poly((x0, y0, x1, y1))
        if element._plot_id in self._cache:
            cache = self._cache[element._plot_id]
            domain, tree, geom_dicts, geom_cache, area_cache = cache
        else:
            if isinstance(element, Polygons):
                geom_dicts = polygons_to_geom_dicts(element)
            elif isinstance(element, Path):
                geom_dicts = path_to_geom_dicts(element)
            geoms = [g['geometry'] for g in geom_dicts]
            tree = STRtree(geoms)
            domain = bounds
            geom_cache, area_cache = ({}, {})
            self._cache.clear()
            cache = (domain, tree, geom_dicts, geom_cache, area_cache)
            self._cache[element._plot_id] = cache
        area = bounds.area
        current_zoom = compute_zoom_level(bounds, domain, self.p.zoom_levels)
        tol = np.sqrt(bounds.area) * self.p.tolerance_factor
        new_geoms, gdict = ([], {})
        for g in tree.query(bounds):
            g = tree.geometries[g] if shapely_v2 else g
            garea = area_cache.get(id(g))
            if garea is None:
                is_poly = 'Polygon' in g.geom_type
                garea = g.area if is_poly else bounds_to_poly(g.bounds).area
                area_cache[id(g)] = garea
            if self.p.display_threshold is not None and garea / area < self.p.display_threshold or not g.intersects(bounds):
                continue
            cache_id = (id(g), current_zoom)
            if cache_id in geom_cache and (not self.p.clip):
                geom_dict = geom_cache[cache_id]
            else:
                if element.vdims:
                    gidx = find_geom(tree._geoms, g)
                    gdict = geom_dicts[gidx]
                g = g.simplify(tol, self.p.preserve_topology)
                if not g:
                    continue
                geom_dict = dict(gdict, geometry=g)
                if self.p.cache:
                    geom_cache[cache_id] = geom_dict
                if self.p.clip:
                    geom_dict = dict(geom_dict, geometry=g.intersection(bounds))
            new_geoms.append(geom_dict)
        return element.clone(new_geoms)