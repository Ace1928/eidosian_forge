import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
class DotConvBase(object):
    """Dot2TeX converter base"""

    def __init__(self, options=None):
        self.color = ''
        self.opacity = None
        try:
            self.template
        except AttributeError:
            self.template = options.get('template', '')
        self.textencoding = options.get('encoding', DEFAULT_TEXTENCODING)
        self.templatevars = {}
        self.body = ''
        if options.get('templatefile', ''):
            self.load_template(options['templatefile'])
        if options.get('template', ''):
            self.template = options['template']
        self.options = options or {}
        if options.get('texpreproc') or options.get('autosize'):
            self.dopreproc = True
        else:
            self.dopreproc = False

    def load_template(self, templatefile):
        try:
            with open(templatefile) as f:
                self.template = f.read()
        except:
            pass

    def convert_file(self, filename):
        """Load dot file and convert"""
        pass

    def start_fig(self):
        return ''

    def end_fig(self):
        return ''

    def draw_ellipse(self, drawop, style=None):
        return ''

    def draw_bezier(self, drawop, style=None):
        return ''

    def draw_polygon(self, drawop, style=None):
        return ''

    def draw_polyline(self, drawop, style=None):
        return ''

    def draw_text(self, drawop, style=None):
        return ''

    def output_node_comment(self, node):
        return '  %% Node: %s\n' % node.name

    def output_edge_comment(self, edge):
        src = edge.get_source()
        dst = edge.get_destination()
        if self.directedgraph:
            edge = '->'
        else:
            edge = '--'
        return '  %% Edge: %s %s %s\n' % (src, edge, dst)

    def set_color(self, node):
        return ''

    def set_style(self, node):
        return ''

    def draw_edge(self, edge):
        return ''

    def start_node(self, node):
        return ''

    def end_node(self, node):
        return ''

    def start_graph(self, graph):
        return ''

    def end_graph(self, graph):
        return ''

    def start_edge(self):
        return ''

    def end_edge(self):
        return ''

    def filter_styles(self, style):
        return style

    def convert_color(self, drawopcolor, pgf=False):
        """Convert color to a format usable by LaTeX and XColor"""
        if drawopcolor.startswith('#'):
            t = list(chunks(drawopcolor[1:], 2))
            if len(t) > 6:
                t = t[0:3]
            rgb = [round(int(n, 16) / 255.0, 2) for n in t]
            if pgf:
                colstr = '{rgb}{%s,%s,%s}' % tuple(rgb[0:3])
                opacity = '1'
                if len(rgb) == 4:
                    opacity = rgb[3]
                return (colstr, opacity)
            else:
                return '[rgb]{%s,%s,%s}' % tuple(rgb[0:3])
        elif len(drawopcolor.split(' ')) == 3 or len(drawopcolor.split(',')) == 3:
            hsb = drawopcolor.split(',')
            if not len(hsb) == 3:
                hsb = drawopcolor.split(' ')
            if pgf:
                return '{hsb}{%s,%s,%s}' % tuple(hsb)
            else:
                return '[hsb]{%s,%s,%s}' % tuple(hsb)
        else:
            drawopcolor = drawopcolor.replace('grey', 'gray')
            drawopcolor = drawopcolor.replace('_', '')
            drawopcolor = drawopcolor.replace(' ', '')
            return drawopcolor

    def do_drawstring(self, drawstring, drawobj, texlbl_name='texlbl', use_drawstring_pos=False):
        """Parse and draw drawsting

        Just a wrapper around do_draw_op.
        """
        drawoperations, stat = parse_drawstring(drawstring)
        return self.do_draw_op(drawoperations, drawobj, stat, texlbl_name, use_drawstring_pos)

    def do_draw_op(self, drawoperations, drawobj, stat, texlbl_name='texlbl', use_drawstring_pos=False):
        """Excecute the operations in drawoperations"""
        s = ''
        for drawop in drawoperations:
            op = drawop[0]
            style = getattr(drawobj, 'style', None)
            if style and (not self.options.get('duplicate')):
                style = self.filter_styles(style)
                styles = [self.styles.get(key.strip(), key.strip()) for key in style.split(',') if key]
                style = ','.join(styles)
            else:
                style = None
            if op in ['e', 'E']:
                s += self.draw_ellipse(drawop, style)
            elif op in ['p', 'P']:
                s += self.draw_polygon(drawop, style)
            elif op == 'L':
                s += self.draw_polyline(drawop, style)
            elif op in ['C', 'c']:
                s += self.set_color(drawop)
            elif op == 'S':
                s += self.set_style(drawop)
            elif op in ['B']:
                s += self.draw_bezier(drawop, style)
            elif op in ['T']:
                text = drawop[5]
                texmode = self.options.get('texmode', 'verbatim')
                if drawobj.attr.get('texmode', ''):
                    texmode = drawobj.attr['texmode']
                if texlbl_name in drawobj.attr:
                    text = drawobj.attr[texlbl_name]
                elif texmode == 'verbatim':
                    text = escape_texchars(text)
                    pass
                elif texmode == 'math':
                    text = '$%s$' % text
                drawop[5] = text
                if self.options.get('alignstr', ''):
                    drawop.append(self.options.get('alignstr'))
                if stat['T'] == 1 and self.options.get('valignmode', 'center') == 'center':
                    drawop[3] = '0'
                    if not use_drawstring_pos:
                        if texlbl_name == 'tailtexlbl':
                            pos = drawobj.attr.get('tail_lp') or drawobj.attr.get('pos')
                        elif texlbl_name == 'headtexlbl':
                            pos = drawobj.attr.get('head_lp') or drawobj.attr.get('pos')
                        else:
                            pos = drawobj.attr.get('lp') or drawobj.attr.get('pos')
                        if pos:
                            coord = pos.split(',')
                            if len(coord) == 2:
                                drawop[1] = coord[0]
                                drawop[2] = coord[1]
                            pass
                lblstyle = drawobj.attr.get('lblstyle')
                exstyle = drawobj.attr.get('exstyle', '')
                if exstyle:
                    if lblstyle:
                        lblstyle += ',' + exstyle
                    else:
                        lblstyle = exstyle
                s += self.draw_text(drawop, lblstyle)
        return s

    def do_nodes(self):
        s = ''
        for node in self.nodes:
            self.currentnode = node
            general_draw_string = node.attr.get('_draw_', '')
            label_string = node.attr.get('_ldraw_', '')
            drawstring = general_draw_string + ' ' + label_string
            if not drawstring.strip():
                continue
            shape = node.attr.get('shape', '')
            if not shape:
                shape = 'ellipse'
            x, y = node.attr.get('pos', '').split(',')
            s += self.output_node_comment(node)
            s += self.start_node(node)
            s += self.do_drawstring(drawstring, node)
            s += self.end_node(node)
        self.body += s

    def get_edge_points(self, edge):
        pos = edge.attr.get('pos')
        if pos:
            segments = pos.split(';')
        else:
            return []
        return_segments = []
        for pos in segments:
            points = pos.split(' ')
            arrow_style = '--'
            i = 0
            if points[i].startswith('s'):
                p = points[0].split(',')
                tmp = '%s,%s' % (p[1], p[2])
                if points[1].startswith('e'):
                    points[2] = tmp
                else:
                    points[1] = tmp
                del points[0]
                arrow_style = '<-'
                i += 1
            if points[0].startswith('e'):
                p = points[0].split(',')
                points.pop()
                points.append('%s,%s' % (p[1], p[2]))
                del points[0]
                arrow_style = '->'
                i += 1
            if i > 1:
                arrow_style = '<->'
            arrow_style = self.get_output_arrow_styles(arrow_style, edge)
            return_segments.append((arrow_style, points))
        return return_segments

    def do_edges(self):
        s = ''
        s += self.set_color(('cC', 'black'))
        for edge in self.edges:
            general_draw_string = edge.attr.get('_draw_', '')
            label_string = edge.attr.get('_ldraw_', '')
            head_arrow_string = edge.attr.get('_hdraw_', '')
            tail_arrow_string = edge.attr.get('_tdraw_', '')
            tail_label_string = edge.attr.get('_tldraw_', '')
            head_label_string = edge.attr.get('_hldraw_', '')
            drawstring = general_draw_string + ' ' + head_arrow_string + ' ' + tail_arrow_string + ' ' + label_string
            drawop, stat = parse_drawstring(drawstring)
            if not drawstring.strip():
                continue
            s += self.output_edge_comment(edge)
            if self.options.get('duplicate'):
                s += self.start_edge()
                s += self.do_draw_op(drawop, edge, stat)
                s += self.do_drawstring(tail_label_string, edge, 'tailtexlbl')
                s += self.do_drawstring(head_label_string, edge, 'headtexlbl')
                s += self.end_edge()
            else:
                s += self.draw_edge(edge)
                s += self.do_drawstring(label_string, edge)
                s += self.do_drawstring(tail_label_string, edge, 'tailtexlbl')
                s += self.do_drawstring(head_label_string, edge, 'headtexlbl')
        self.body += s

    def do_graph(self):
        general_draw_string = self.graph.attr.get('_draw_', '')
        label_string = self.graph.attr.get('_ldraw_', '')
        if general_draw_string.startswith('c 5 -white C 5 -white') and (not self.graph.attr.get('style')):
            general_draw_string = ''
        if getattr(self.graph, '_draw_', None):
            general_draw_string = 'c 5 -black ' + general_draw_string
            pass
        drawstring = general_draw_string + ' ' + label_string
        if drawstring.strip():
            s = self.start_graph(self.graph)
            g = self.do_drawstring(drawstring, self.graph)
            e = self.end_graph(self.graph)
            if g.strip():
                self.body += s + g + e

    def set_options(self):
        self.options['alignstr'] = self.options.get('alignstr', '') or getattr(self.main_graph, 'd2talignstr', '')
        self.options['valignmode'] = getattr(self.main_graph, 'd2tvalignmode', '') or self.options.get('valignmode', 'center')

    def convert(self, dotdata):
        log.debug('Start conversion')
        main_graph = parse_dot_data(dotdata)
        if not self.dopreproc and (not hasattr(main_graph, 'xdotversion')):
            if not (dotdata.find('_draw_') > 0 or dotdata.find('_ldraw_') > 0):
                log.info('Trying to create xdotdata')
                tmpdata = create_xdot(dotdata, self.options.get('prog', 'dot'), options=self.options.get('progoptions', ''))
                if tmpdata is None or not tmpdata.strip():
                    log.error('Failed to create xdotdata. Is Graphviz installed?')
                    sys.exit(1)
                log.debug('xdotdata:\n' + str(tmpdata))
                main_graph = parse_dot_data(tmpdata)
                log.debug('dotparsing graph:\n' + str(main_graph))
            else:
                pass
        self.main_graph = main_graph
        self.pencolor = ''
        self.fillcolor = ''
        self.linewidth = 1
        self.directedgraph = main_graph.directed
        if self.dopreproc:
            return self.do_preview_preproc()
        dstring = self.main_graph.attr.get('_draw_', '')
        if dstring:
            self.main_graph.attr['_draw_'] = ''
        self.set_options()
        graphlist = get_graphlist(self.main_graph, [])
        self.body += self.start_fig()
        for graph in graphlist:
            self.graph = graph
            self.do_graph()
        if True:
            self.nodes = list(main_graph.allnodes)
            self.edges = list(main_graph.alledges)
            if not self.options.get('switchdraworder'):
                self.do_edges()
                self.do_nodes()
            else:
                self.do_nodes()
                self.do_edges()
        self.body += self.end_fig()
        return self.output()

    def clean_template(self, template):
        """Remove preprocsection or outputsection"""
        if not self.dopreproc and self.options.get('codeonly'):
            r = re.compile('<<startcodeonlysection>>(.*?)<<endcodeonlysection>>', re.DOTALL | re.MULTILINE)
            m = r.search(template)
            if m:
                return m.group(1).strip()
        if not self.dopreproc and self.options.get('figonly'):
            r = re.compile('<<start_figonlysection>>(.*?)<<end_figonlysection>>', re.DOTALL | re.MULTILINE)
            m = r.search(template)
            if m:
                return m.group(1)
            r = re.compile('<<startfigonlysection>>(.*?)<<endfigonlysection>>', re.DOTALL | re.MULTILINE)
            m = r.search(template)
            if m:
                return m.group(1)
        if self.dopreproc:
            r = re.compile('<<startoutputsection>>.*?<<endoutputsection>>', re.DOTALL | re.MULTILINE)
        else:
            r = re.compile('<<startpreprocsection>>.*?<<endpreprocsection>>', re.DOTALL | re.MULTILINE)
        r2 = re.compile('<<start_figonlysection>>.*?<<end_figonlysection>>', re.DOTALL | re.MULTILINE)
        tmp = r2.sub('', template)
        r2 = re.compile('<<startcodeonlysection>>.*?<<endcodeonlysection>>', re.DOTALL | re.MULTILINE)
        tmp = r2.sub('', tmp)
        return r.sub('', tmp)

    def init_template_vars(self):
        variables = {}
        bbstr = self.main_graph.attr.get('bb', '')
        if bbstr:
            bb = bbstr.split(',')
            variables['<<bbox>>'] = '(%sbp,%sbp)(%sbp,%sbp)\n' % (smart_float(bb[0]), smart_float(bb[1]), smart_float(bb[2]), smart_float(bb[3]))
            variables['<<bbox.x0>>'] = bb[0]
            variables['<<bbox.y0>>'] = bb[1]
            variables['<<bbox.x1>>'] = bb[2]
            variables['<<bbox.y1>>'] = bb[3]
        variables['<<figcode>>'] = self.body.strip()
        variables['<<drawcommands>>'] = self.body.strip()
        variables['<<textencoding>>'] = self.textencoding
        docpreamble = self.options.get('docpreamble', '') or getattr(self.main_graph, 'd2tdocpreamble', '')
        variables['<<docpreamble>>'] = docpreamble
        variables['<<figpreamble>>'] = self.options.get('figpreamble', '') or getattr(self.main_graph, 'd2tfigpreamble', '%')
        variables['<<figpostamble>>'] = self.options.get('figpostamble', '') or getattr(self.main_graph, 'd2tfigpostamble', '')
        variables['<<graphstyle>>'] = self.options.get('graphstyle', '') or getattr(self.main_graph, 'd2tgraphstyle', '')
        variables['<<margin>>'] = self.options.get('margin', '0pt')
        variables['<<startpreprocsection>>'] = variables['<<endpreprocsection>>'] = ''
        variables['<<startoutputsection>>'] = variables['<<endoutputsection>>'] = ''
        if self.options.get('gvcols'):
            variables['<<gvcols>>'] = '\\input{gvcols.tex}'
        else:
            variables['<<gvcols>>'] = ''
        self.templatevars = variables

    def output(self):
        self.init_template_vars()
        template = self.clean_template(self.template)
        code = replace_tags(template, self.templatevars, self.templatevars)
        return code

    def get_label(self, drawobj, label_attribute='label', tex_label_attribute='texlbl'):
        text = ''
        texmode = self.options.get('texmode', 'verbatim')
        if getattr(drawobj, 'texmode', ''):
            texmode = drawobj.texmode
        text = getattr(drawobj, label_attribute, None)
        if text is None or text.strip() == '\\N':
            if not isinstance(drawobj, dotparsing.DotEdge):
                text = getattr(drawobj, 'name', None) or getattr(drawobj, 'graph_name', '')
                text = text.replace('\\\\', '\\')
            else:
                text = ''
        elif text.strip() == '\\N':
            text = ''
        else:
            text = text.replace('\\\\', '\\')
        if getattr(drawobj, tex_label_attribute, ''):
            text = drawobj.texlbl
        elif texmode == 'verbatim':
            text = escape_texchars(text)
            pass
        elif texmode == 'math':
            text = '$%s$' % text
        return text

    def get_node_preproc_code(self, node):
        return node.attr.get('texlbl', '')

    def get_edge_preproc_code(self, edge, attribute='texlbl'):
        return edge.attr.get(attribute, '')

    def get_graph_preproc_code(self, graph):
        return graph.attr.get('texlbl', '')

    def get_margins(self, element):
        """Return element margins"""
        margins = element.attr.get('margin')
        if margins:
            margins = margins.split(',')
            if len(margins) == 1:
                xmargin = ymargin = float(margins[0])
            else:
                xmargin = float(margins[0])
                ymargin = float(margins[1])
        elif isinstance(element, dotparsing.DotEdge):
            xmargin = DEFAULT_EDGELABEL_XMARGIN
            ymargin = DEFAULT_EDGELABEL_YMARGIN
        else:
            xmargin = DEFAULT_LABEL_XMARGIN
            ymargin = DEFAULT_LABEL_YMARGIN
        return (xmargin, ymargin)

    def do_preview_preproc(self):
        self.init_template_vars()
        template = self.clean_template(self.template)
        template = replace_tags(template, self.templatevars, self.templatevars)
        pp = TeXDimProc(template, self.options)
        usednodes = {}
        usededges = {}
        usedgraphs = {}
        counter = 0
        for node in self.main_graph.allnodes:
            name = node.name
            if node.attr.get('fixedsize', '') == 'true' or node.attr.get('style', '') in ['invis', 'invisible']:
                continue
            if node.attr.get('shape', '') == 'record':
                log.warning('Record nodes not supported in preprocessing mode: %s', name)
                continue
            texlbl = self.get_label(node)
            if texlbl:
                node.attr['texlbl'] = texlbl
                code = self.get_node_preproc_code(node)
                pp.add_snippet(name, code)
            usednodes[name] = node
        for edge in dotparsing.flatten(self.main_graph.alledges):
            if not edge.attr.get('label') and (not edge.attr.get('texlbl')) and (not edge.attr.get('headlabel')) and (not edge.attr.get('taillabel')):
                continue
            name = edge.src.name + edge.dst.name + str(counter)
            if is_multiline_label(edge):
                continue
            label = self.get_label(edge)
            headlabel = self.get_label(edge, 'headlabel', 'headtexlbl')
            taillabel = self.get_label(edge, 'taillabel', 'tailtexlbl')
            if label:
                name = edge.src.name + edge.dst.name + str(counter)
                edge.attr['texlbl'] = label
                code = self.get_edge_preproc_code(edge)
                pp.add_snippet(name, code)
            if headlabel:
                headlabel_name = name + 'headlabel'
                edge.attr['headtexlbl'] = headlabel
                code = self.get_edge_preproc_code(edge, 'headtexlbl')
                pp.add_snippet(headlabel_name, code)
            if taillabel:
                taillabel_name = name + 'taillabel'
                edge.attr['tailtexlbl'] = taillabel
                code = self.get_edge_preproc_code(edge, 'tailtexlbl')
                pp.add_snippet(taillabel_name, code)
            counter += 1
            usededges[name] = edge
        for graph in self.main_graph.allgraphs:
            if not graph.attr.get('label') and (not graph.attr.get('texlbl')):
                continue
            name = graph.name + str(counter)
            counter += 1
            label = self.get_label(graph)
            graph.attr['texlbl'] = label
            code = self.get_graph_preproc_code(graph)
            pp.add_snippet(name, code)
            usedgraphs[name] = graph
        ok = pp.process()
        if not ok:
            errormsg = 'Failed to preprocess the graph.\nIs the preview LaTeX package installed? ((Debian package preview-latex-style)\nTo see what happened, run dot2tex with the --debug option.\n'
            log.error(errormsg)
            sys.exit(1)
        for name, item in usednodes.items():
            if not item.attr.get('texlbl'):
                continue
            node = item
            hp, dp, wt = pp.texdims[name]
            if self.options.get('rawdim'):
                node.attr['width'] = wt
                node.attr['height'] = hp + dp
                node.attr['label'] = ' '
                node.attr['fixedsize'] = 'true'
                self.main_graph.allitems.append(node)
                continue
            xmargin, ymargin = self.get_margins(node)
            ht = hp + dp
            minwidth = float(item.attr.get('width') or DEFAULT_NODE_WIDTH)
            minheight = float(item.attr.get('height') or DEFAULT_NODE_HEIGHT)
            if self.options.get('nominsize'):
                width = wt + 2 * xmargin
                height = ht + 2 * ymargin
            else:
                if wt + 2 * xmargin < minwidth:
                    width = minwidth
                else:
                    width = wt + 2 * xmargin
                height = ht
                if hp + dp + 2 * ymargin < minheight:
                    height = minheight
                else:
                    height = ht + 2 * ymargin
            if item.attr.get('shape', '') in ['circle', 'Msquare', 'doublecircle', 'Mcircle']:
                if wt < height and width < height:
                    width = height
                else:
                    height = width
            node.attr['width'] = width
            node.attr['height'] = height
            node.attr['label'] = ' '
            node.attr['fixedsize'] = 'true'
            self.main_graph.allitems.append(node)
        for name, item in usededges.items():
            edge = item
            hp, dp, wt = pp.texdims[name]
            xmargin, ymargin = self.get_margins(edge)
            labelcode = '<<<table border="0" cellborder="0" cellpadding="0"><tr><td fixedsize="true" width="%s" height="%s">a</td></tr></table>>>'
            if 'texlbl' in edge.attr:
                edge.attr['label'] = labelcode % ((wt + 2 * xmargin) * 72, (hp + dp + 2 * ymargin) * 72)
            if 'tailtexlbl' in edge.attr:
                hp, dp, wt = pp.texdims[name + 'taillabel']
                edge.attr['taillabel'] = labelcode % ((wt + 2 * xmargin) * 72, (hp + dp + 2 * ymargin) * 72)
            if 'headtexlbl' in edge.attr:
                hp, dp, wt = pp.texdims[name + 'headlabel']
                edge.attr['headlabel'] = labelcode % ((wt + 2 * xmargin) * 72, (hp + dp + 2 * ymargin) * 72)
        for name, item in usedgraphs.items():
            graph = item
            hp, dp, wt = pp.texdims[name]
            xmargin, ymargin = self.get_margins(graph)
            labelcode = '<<<table border="0" cellborder="0" cellpadding="0"><tr><td fixedsize="true" width="%s" height="%s">a</td></tr></table>>>'
            graph.attr['label'] = labelcode % ((wt + 2 * xmargin) * 72, (hp + dp + 2 * ymargin) * 72)
        self.main_graph.attr['d2toutputformat'] = self.options.get('format', DEFAULT_OUTPUT_FORMAT)
        graphcode = str(self.main_graph)
        graphcode = graphcode.replace('<<<', '<<')
        graphcode = graphcode.replace('>>>', '>>')
        return graphcode

    def get_output_arrow_styles(self, arrow_style, edge):
        return arrow_style