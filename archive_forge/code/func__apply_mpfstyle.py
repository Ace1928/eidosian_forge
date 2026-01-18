import matplotlib.pyplot as plt
import copy
import pprint
import os.path as path
from   mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from   mplfinance._styledata      import _styles
from   mplfinance._helpers        import _mpf_is_color_like
def _apply_mpfstyle(style):
    plt.style.use('default')
    if style['base_mpl_style'] == 'seaborn-darkgrid':
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            style['base_mpl_style'] = 'seaborn-v0_8-darkgrid'
        except:
            plt.style.use(style['base_mpl_style'])
    elif style['base_mpl_style'] is not None:
        plt.style.use(style['base_mpl_style'])
    if style['rc'] is not None:
        plt.rcParams.update(style['rc'])
    if style['facecolor'] is not None:
        plt.rcParams.update({'axes.facecolor': style['facecolor']})
    if 'edgecolor' in style and style['edgecolor'] is not None:
        plt.rcParams.update({'axes.edgecolor': style['edgecolor']})
    if 'figcolor' in style and style['figcolor'] is not None:
        plt.rcParams.update({'figure.facecolor': style['figcolor']})
        plt.rcParams.update({'savefig.facecolor': style['figcolor']})
    explicit_grid = False
    if style['gridcolor'] is not None:
        explicit_grid = True
        plt.rcParams.update({'grid.color': style['gridcolor']})
    if style['gridstyle'] is not None:
        explicit_grid = True
        plt.rcParams.update({'grid.linestyle': style['gridstyle']})
    plt.rcParams.update({'axes.grid.axis': 'both'})
    if 'gridaxis' in style and style['gridaxis'] is not None:
        gax = style['gridaxis']
        explicit_grid = True
        if gax == 'horizontal'[0:len(gax)]:
            plt.rcParams.update({'axes.grid.axis': 'y'})
        elif gax == 'vertical'[0:len(gax)]:
            plt.rcParams.update({'axes.grid.axis': 'x'})
    if explicit_grid:
        plt.rcParams.update({'axes.grid': True})