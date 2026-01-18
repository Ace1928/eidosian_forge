from freetype.ft_enums.ft_render_modes import *
def _FT_LOAD_TARGET_(x):
    return (x & 15) << 16