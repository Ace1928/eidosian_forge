from plotly.basedatatypes import BaseLayoutType as _BaseLayoutType
import copy as _copy
@property
def _subplotid_validators(self):
    """
        dict of validator classes for each subplot type

        Returns
        -------
        dict
        """
    from plotly.validators.layout import ColoraxisValidator, GeoValidator, LegendValidator, MapboxValidator, PolarValidator, SceneValidator, SmithValidator, TernaryValidator, XaxisValidator, YaxisValidator
    return {'coloraxis': ColoraxisValidator, 'geo': GeoValidator, 'legend': LegendValidator, 'mapbox': MapboxValidator, 'polar': PolarValidator, 'scene': SceneValidator, 'smith': SmithValidator, 'ternary': TernaryValidator, 'xaxis': XaxisValidator, 'yaxis': YaxisValidator}