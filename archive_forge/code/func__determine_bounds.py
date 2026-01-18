import numpy as np
import cartopy.crs as ccrs
def _determine_bounds(x_coords, y_coords, source_cs):
    bounds = dict(x=[])
    half_px = abs(np.diff(x_coords[:2])).max() / 2.0
    if (hasattr(source_cs, 'is_geodetic') and source_cs.is_geodetic() or isinstance(source_cs, ccrs.PlateCarree)) and x_coords.max() > 180:
        if x_coords.min() < 180:
            bounds['x'].append([x_coords.min() - half_px, 180])
            bounds['x'].append([-180, x_coords.max() - 360 + half_px])
        else:
            bounds['x'].append([x_coords.min() - 180 - half_px, x_coords.max() - 180 + half_px])
    else:
        bounds['x'].append([x_coords.min() - half_px, x_coords.max() + half_px])
    bounds['y'] = [y_coords.min(), y_coords.max()]
    return bounds