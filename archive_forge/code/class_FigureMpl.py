from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.images import Figure, Image
import os
from os.path import relpath
from pathlib import PurePath, Path
import shutil
from sphinx.errors import ExtensionError
import matplotlib
class FigureMpl(Figure):
    """
    Implements a directive to allow an optional hidpi image.

    Meant to be used with the *plot_srcset* configuration option in conf.py,
    and gets set in the TEMPLATE of plot_directive.py

    e.g.::

        .. figure-mpl:: plot_directive/some_plots-1.png
            :alt: bar
            :srcset: plot_directive/some_plots-1.png,
                     plot_directive/some_plots-1.2x.png 2.00x
            :class: plot-directive

    The resulting html (at ``some_plots.html``) is::

        <img src="sphx_glr_bar_001_hidpi.png"
            srcset="_images/some_plot-1.png,
                    _images/some_plots-1.2x.png 2.00x",
            alt="bar"
            class="plot_directive" />

    Note that the handling of subdirectories is different than that used by the sphinx
    figure directive::

        .. figure-mpl:: plot_directive/nestedpage/index-1.png
            :alt: bar
            :srcset: plot_directive/nestedpage/index-1.png
                     plot_directive/nestedpage/index-1.2x.png 2.00x
            :class: plot_directive

    The resulting html (at ``nestedpage/index.html``)::

        <img src="../_images/nestedpage-index-1.png"
            srcset="../_images/nestedpage-index-1.png,
                    ../_images/_images/nestedpage-index-1.2x.png 2.00x",
            alt="bar"
            class="sphx-glr-single-img" />

    where the subdirectory is included in the image name for uniqueness.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 2
    final_argument_whitespace = False
    option_spec = {'alt': directives.unchanged, 'height': directives.length_or_unitless, 'width': directives.length_or_percentage_or_unitless, 'scale': directives.nonnegative_int, 'align': Image.align, 'class': directives.class_option, 'caption': directives.unchanged, 'srcset': directives.unchanged}

    def run(self):
        image_node = figmplnode()
        imagenm = self.arguments[0]
        image_node['alt'] = self.options.get('alt', '')
        image_node['align'] = self.options.get('align', None)
        image_node['class'] = self.options.get('class', None)
        image_node['width'] = self.options.get('width', None)
        image_node['height'] = self.options.get('height', None)
        image_node['scale'] = self.options.get('scale', None)
        image_node['caption'] = self.options.get('caption', None)
        image_node['uri'] = imagenm
        image_node['srcset'] = self.options.get('srcset', None)
        return [image_node]