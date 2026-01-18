from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
class XypicDiagramDrawer:
    """
    Given a :class:`~.Diagram` and the corresponding
    :class:`DiagramGrid`, produces the Xy-pic representation of the
    diagram.

    The most important method in this class is ``draw``.  Consider the
    following triangle diagram:

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import DiagramGrid, XypicDiagramDrawer
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g], {g * f: "unique"})

    To draw this diagram, its objects need to be laid out with a
    :class:`DiagramGrid`::

    >>> grid = DiagramGrid(diagram)

    Finally, the drawing:

    >>> drawer = XypicDiagramDrawer()
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar[d]_{g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\
    C &
    }

    For further details see the docstring of this method.

    To control the appearance of the arrows, formatters are used.  The
    dictionary ``arrow_formatters`` maps morphisms to formatter
    functions.  A formatter is accepts an
    :class:`ArrowStringDescription` and is allowed to modify any of
    the arrow properties exposed thereby.  For example, to have all
    morphisms with the property ``unique`` appear as dashed arrows,
    and to have their names prepended with `\\exists !`, the following
    should be done:

    >>> def formatter(astr):
    ...   astr.label = r"\\exists !" + astr.label
    ...   astr.arrow_style = "{-->}"
    >>> drawer.arrow_formatters["unique"] = formatter
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar@{-->}[d]_{\\exists !g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\
    C &
    }

    To modify the appearance of all arrows in the diagram, set
    ``default_arrow_formatter``.  For example, to place all morphism
    labels a little bit farther from the arrow head so that they look
    more centred, do as follows:

    >>> def default_formatter(astr):
    ...   astr.label_displacement = "(0.45)"
    >>> drawer.default_arrow_formatter = default_formatter
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar@{-->}[d]_(0.45){\\exists !g\\circ f} \\ar[r]^(0.45){f} & B \\ar[ld]^(0.45){g} \\\\
    C &
    }

    In some diagrams some morphisms are drawn as curved arrows.
    Consider the following diagram:

    >>> D = Object("D")
    >>> E = Object("E")
    >>> h = NamedMorphism(D, A, "h")
    >>> k = NamedMorphism(D, B, "k")
    >>> diagram = Diagram([f, g, h, k])
    >>> grid = DiagramGrid(diagram)
    >>> drawer = XypicDiagramDrawer()
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar[r]_{f} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_3mm/[ll]_{h} \\\\
    & C &
    }

    To control how far the morphisms are curved by default, one can
    use the ``unit`` and ``default_curving_amount`` attributes:

    >>> drawer.unit = "cm"
    >>> drawer.default_curving_amount = 1
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar[r]_{f} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_1cm/[ll]_{h} \\\\
    & C &
    }

    In some diagrams, there are multiple curved morphisms between the
    same two objects.  To control by how much the curving changes
    between two such successive morphisms, use
    ``default_curving_step``:

    >>> drawer.default_curving_step = 1
    >>> h1 = NamedMorphism(A, D, "h1")
    >>> diagram = Diagram([f, g, h, k, h1])
    >>> grid = DiagramGrid(diagram)
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar[r]_{f} \\ar@/^1cm/[rr]^{h_{1}} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_2cm/[ll]_{h} \\\\
    & C &
    }

    The default value of ``default_curving_step`` is 4 units.

    See Also
    ========

    draw, ArrowStringDescription
    """

    def __init__(self):
        self.unit = 'mm'
        self.default_curving_amount = 3
        self.default_curving_step = 4
        self.arrow_formatters = {}
        self.default_arrow_formatter = None

    @staticmethod
    def _process_loop_morphism(i, j, grid, morphisms_str_info, object_coords):
        """
        Produces the information required for constructing the string
        representation of a loop morphism.  This function is invoked
        from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
        curving = ''
        label_pos = '^'
        looping_start = ''
        looping_end = ''
        quadrant = [0, 0, 0, 0]
        obj = grid[i, j]
        for m, m_str_info in morphisms_str_info.items():
            if m.domain == obj and m.codomain == obj:
                l_s, l_e = (m_str_info.looping_start, m_str_info.looping_end)
                if (l_s, l_e) == ('r', 'u'):
                    quadrant[0] += 1
                elif (l_s, l_e) == ('u', 'l'):
                    quadrant[1] += 1
                elif (l_s, l_e) == ('l', 'd'):
                    quadrant[2] += 1
                elif (l_s, l_e) == ('d', 'r'):
                    quadrant[3] += 1
                continue
            if m.domain == obj:
                end_i, end_j = object_coords[m.codomain]
                goes_out = True
            elif m.codomain == obj:
                end_i, end_j = object_coords[m.domain]
                goes_out = False
            else:
                continue
            d_i = end_i - i
            d_j = end_j - j
            m_curving = m_str_info.curving
            if d_i != 0 and d_j != 0:
                if d_i > 0 and d_j > 0:
                    quadrant[0] += 1
                elif d_i > 0 and d_j < 0:
                    quadrant[1] += 1
                elif d_i < 0 and d_j < 0:
                    quadrant[2] += 1
                elif d_i < 0 and d_j > 0:
                    quadrant[3] += 1
            elif d_i == 0:
                if d_j > 0:
                    if goes_out:
                        upper_quadrant = 0
                        lower_quadrant = 3
                    else:
                        upper_quadrant = 3
                        lower_quadrant = 0
                elif goes_out:
                    upper_quadrant = 2
                    lower_quadrant = 1
                else:
                    upper_quadrant = 1
                    lower_quadrant = 2
                if m_curving:
                    if m_curving == '^':
                        quadrant[upper_quadrant] += 1
                    elif m_curving == '_':
                        quadrant[lower_quadrant] += 1
                else:
                    quadrant[upper_quadrant] += 1
                    quadrant[lower_quadrant] += 1
            elif d_j == 0:
                if d_i < 0:
                    if goes_out:
                        left_quadrant = 1
                        right_quadrant = 0
                    else:
                        left_quadrant = 0
                        right_quadrant = 1
                elif goes_out:
                    left_quadrant = 3
                    right_quadrant = 2
                else:
                    left_quadrant = 2
                    right_quadrant = 3
                if m_curving:
                    if m_curving == '^':
                        quadrant[left_quadrant] += 1
                    elif m_curving == '_':
                        quadrant[right_quadrant] += 1
                else:
                    quadrant[left_quadrant] += 1
                    quadrant[right_quadrant] += 1
        freest_quadrant = 0
        for i in range(4):
            if quadrant[i] < quadrant[freest_quadrant]:
                freest_quadrant = i
        looping_start, looping_end = [('r', 'u'), ('u', 'l'), ('l', 'd'), ('d', 'r')][freest_quadrant]
        return (curving, label_pos, looping_start, looping_end)

    @staticmethod
    def _process_horizontal_morphism(i, j, target_j, grid, morphisms_str_info, object_coords):
        """
        Produces the information required for constructing the string
        representation of a horizontal morphism.  This function is
        invoked from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
        backwards = False
        start = j
        end = target_j
        if end < start:
            start, end = (end, start)
            backwards = True
        up = []
        down = []
        straight_horizontal = []
        for k in range(start + 1, end):
            obj = grid[i, k]
            if not obj:
                continue
            for m in morphisms_str_info:
                if m.domain == obj:
                    end_i, end_j = object_coords[m.codomain]
                elif m.codomain == obj:
                    end_i, end_j = object_coords[m.domain]
                else:
                    continue
                if end_i > i:
                    down.append(m)
                elif end_i < i:
                    up.append(m)
                elif not morphisms_str_info[m].curving:
                    straight_horizontal.append(m)
        if len(up) < len(down):
            if backwards:
                curving = '_'
                label_pos = '_'
            else:
                curving = '^'
                label_pos = '^'
            for m in straight_horizontal:
                i1, j1 = object_coords[m.domain]
                i2, j2 = object_coords[m.codomain]
                m_str_info = morphisms_str_info[m]
                if j1 < j2:
                    m_str_info.label_position = '_'
                else:
                    m_str_info.label_position = '^'
                m_str_info.forced_label_position = True
        else:
            if backwards:
                curving = '^'
                label_pos = '^'
            else:
                curving = '_'
                label_pos = '_'
            for m in straight_horizontal:
                i1, j1 = object_coords[m.domain]
                i2, j2 = object_coords[m.codomain]
                m_str_info = morphisms_str_info[m]
                if j1 < j2:
                    m_str_info.label_position = '^'
                else:
                    m_str_info.label_position = '_'
                m_str_info.forced_label_position = True
        return (curving, label_pos)

    @staticmethod
    def _process_vertical_morphism(i, j, target_i, grid, morphisms_str_info, object_coords):
        """
        Produces the information required for constructing the string
        representation of a vertical morphism.  This function is
        invoked from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
        backwards = False
        start = i
        end = target_i
        if end < start:
            start, end = (end, start)
            backwards = True
        left = []
        right = []
        straight_vertical = []
        for k in range(start + 1, end):
            obj = grid[k, j]
            if not obj:
                continue
            for m in morphisms_str_info:
                if m.domain == obj:
                    end_i, end_j = object_coords[m.codomain]
                elif m.codomain == obj:
                    end_i, end_j = object_coords[m.domain]
                else:
                    continue
                if end_j > j:
                    right.append(m)
                elif end_j < j:
                    left.append(m)
                elif not morphisms_str_info[m].curving:
                    straight_vertical.append(m)
        if len(left) < len(right):
            if backwards:
                curving = '^'
                label_pos = '^'
            else:
                curving = '_'
                label_pos = '_'
            for m in straight_vertical:
                i1, j1 = object_coords[m.domain]
                i2, j2 = object_coords[m.codomain]
                m_str_info = morphisms_str_info[m]
                if i1 < i2:
                    m_str_info.label_position = '^'
                else:
                    m_str_info.label_position = '_'
                m_str_info.forced_label_position = True
        else:
            if backwards:
                curving = '_'
                label_pos = '_'
            else:
                curving = '^'
                label_pos = '^'
            for m in straight_vertical:
                i1, j1 = object_coords[m.domain]
                i2, j2 = object_coords[m.codomain]
                m_str_info = morphisms_str_info[m]
                if i1 < i2:
                    m_str_info.label_position = '_'
                else:
                    m_str_info.label_position = '^'
                m_str_info.forced_label_position = True
        return (curving, label_pos)

    def _process_morphism(self, diagram, grid, morphism, object_coords, morphisms, morphisms_str_info):
        """
        Given the required information, produces the string
        representation of ``morphism``.
        """

        def repeat_string_cond(times, str_gt, str_lt):
            """
            If ``times > 0``, repeats ``str_gt`` ``times`` times.
            Otherwise, repeats ``str_lt`` ``-times`` times.
            """
            if times > 0:
                return str_gt * times
            else:
                return str_lt * -times

        def count_morphisms_undirected(A, B):
            """
            Counts how many processed morphisms there are between the
            two supplied objects.
            """
            return len([m for m in morphisms_str_info if {m.domain, m.codomain} == {A, B}])

        def count_morphisms_filtered(dom, cod, curving):
            """
            Counts the processed morphisms which go out of ``dom``
            into ``cod`` with curving ``curving``.
            """
            return len([m for m, m_str_info in morphisms_str_info.items() if (m.domain, m.codomain) == (dom, cod) and m_str_info.curving == curving])
        i, j = object_coords[morphism.domain]
        target_i, target_j = object_coords[morphism.codomain]
        delta_i = target_i - i
        delta_j = target_j - j
        vertical_direction = repeat_string_cond(delta_i, 'd', 'u')
        horizontal_direction = repeat_string_cond(delta_j, 'r', 'l')
        curving = ''
        label_pos = '^'
        looping_start = ''
        looping_end = ''
        if delta_i == 0 and delta_j == 0:
            curving, label_pos, looping_start, looping_end = XypicDiagramDrawer._process_loop_morphism(i, j, grid, morphisms_str_info, object_coords)
        elif delta_i == 0 and abs(j - target_j) > 1:
            curving, label_pos = XypicDiagramDrawer._process_horizontal_morphism(i, j, target_j, grid, morphisms_str_info, object_coords)
        elif delta_j == 0 and abs(i - target_i) > 1:
            curving, label_pos = XypicDiagramDrawer._process_vertical_morphism(i, j, target_i, grid, morphisms_str_info, object_coords)
        count = count_morphisms_undirected(morphism.domain, morphism.codomain)
        curving_amount = ''
        if curving:
            curving_amount = self.default_curving_amount + count * self.default_curving_step
        elif count:
            curving = '^'
            filtered_morphisms = count_morphisms_filtered(morphism.domain, morphism.codomain, curving)
            curving_amount = self.default_curving_amount + filtered_morphisms * self.default_curving_step
        morphism_name = ''
        if isinstance(morphism, IdentityMorphism):
            morphism_name = 'id_{%s}' + latex(grid[i, j])
        elif isinstance(morphism, CompositeMorphism):
            component_names = [latex(Symbol(component.name)) for component in morphism.components]
            component_names.reverse()
            morphism_name = '\\circ '.join(component_names)
        elif isinstance(morphism, NamedMorphism):
            morphism_name = latex(Symbol(morphism.name))
        return ArrowStringDescription(self.unit, curving, curving_amount, looping_start, looping_end, horizontal_direction, vertical_direction, label_pos, morphism_name)

    @staticmethod
    def _check_free_space_horizontal(dom_i, dom_j, cod_j, grid):
        """
        For a horizontal morphism, checks whether there is free space
        (i.e., space not occupied by any objects) above the morphism
        or below it.
        """
        if dom_j < cod_j:
            start, end = (dom_j, cod_j)
            backwards = False
        else:
            start, end = (cod_j, dom_j)
            backwards = True
        if dom_i == 0:
            free_up = True
        else:
            free_up = all((grid[dom_i - 1, j] for j in range(start, end + 1)))
        if dom_i == grid.height - 1:
            free_down = True
        else:
            free_down = not any((grid[dom_i + 1, j] for j in range(start, end + 1)))
        return (free_up, free_down, backwards)

    @staticmethod
    def _check_free_space_vertical(dom_i, cod_i, dom_j, grid):
        """
        For a vertical morphism, checks whether there is free space
        (i.e., space not occupied by any objects) to the left of the
        morphism or to the right of it.
        """
        if dom_i < cod_i:
            start, end = (dom_i, cod_i)
            backwards = False
        else:
            start, end = (cod_i, dom_i)
            backwards = True
        if dom_j == 0:
            free_left = True
        else:
            free_left = not any((grid[i, dom_j - 1] for i in range(start, end + 1)))
        if dom_j == grid.width - 1:
            free_right = True
        else:
            free_right = not any((grid[i, dom_j + 1] for i in range(start, end + 1)))
        return (free_left, free_right, backwards)

    @staticmethod
    def _check_free_space_diagonal(dom_i, cod_i, dom_j, cod_j, grid):
        """
        For a diagonal morphism, checks whether there is free space
        (i.e., space not occupied by any objects) above the morphism
        or below it.
        """

        def abs_xrange(start, end):
            if start < end:
                return range(start, end + 1)
            else:
                return range(end, start + 1)
        if dom_i < cod_i and dom_j < cod_j:
            start_i, start_j = (dom_i, dom_j)
            end_i, end_j = (cod_i, cod_j)
            backwards = False
        elif dom_i > cod_i and dom_j > cod_j:
            start_i, start_j = (cod_i, cod_j)
            end_i, end_j = (dom_i, dom_j)
            backwards = True
        if dom_i < cod_i and dom_j > cod_j:
            start_i, start_j = (dom_i, dom_j)
            end_i, end_j = (cod_i, cod_j)
            backwards = True
        elif dom_i > cod_i and dom_j < cod_j:
            start_i, start_j = (cod_i, cod_j)
            end_i, end_j = (dom_i, dom_j)
            backwards = False
        alpha = float(end_i - start_i) / (end_j - start_j)
        free_up = True
        free_down = True
        for i in abs_xrange(start_i, end_i):
            if not free_up and (not free_down):
                break
            for j in abs_xrange(start_j, end_j):
                if not free_up and (not free_down):
                    break
                if (i, j) == (start_i, start_j):
                    continue
                if j == start_j:
                    alpha1 = 'inf'
                else:
                    alpha1 = float(i - start_i) / (j - start_j)
                if grid[i, j]:
                    if alpha1 == 'inf' or abs(alpha1) > abs(alpha):
                        free_down = False
                    elif abs(alpha1) < abs(alpha):
                        free_up = False
        return (free_up, free_down, backwards)

    def _push_labels_out(self, morphisms_str_info, grid, object_coords):
        """
        For all straight morphisms which form the visual boundary of
        the laid out diagram, puts their labels on their outer sides.
        """

        def set_label_position(free1, free2, pos1, pos2, backwards, m_str_info):
            """
            Given the information about room available to one side and
            to the other side of a morphism (``free1`` and ``free2``),
            sets the position of the morphism label in such a way that
            it is on the freer side.  This latter operations involves
            choice between ``pos1`` and ``pos2``, taking ``backwards``
            in consideration.

            Thus this function will do nothing if either both ``free1
            == True`` and ``free2 == True`` or both ``free1 == False``
            and ``free2 == False``.  In either case, choosing one side
            over the other presents no advantage.
            """
            if backwards:
                pos1, pos2 = (pos2, pos1)
            if free1 and (not free2):
                m_str_info.label_position = pos1
            elif free2 and (not free1):
                m_str_info.label_position = pos2
        for m, m_str_info in morphisms_str_info.items():
            if m_str_info.curving or m_str_info.forced_label_position:
                continue
            if m.domain == m.codomain:
                continue
            dom_i, dom_j = object_coords[m.domain]
            cod_i, cod_j = object_coords[m.codomain]
            if dom_i == cod_i:
                free_up, free_down, backwards = XypicDiagramDrawer._check_free_space_horizontal(dom_i, dom_j, cod_j, grid)
                set_label_position(free_up, free_down, '^', '_', backwards, m_str_info)
            elif dom_j == cod_j:
                free_left, free_right, backwards = XypicDiagramDrawer._check_free_space_vertical(dom_i, cod_i, dom_j, grid)
                set_label_position(free_left, free_right, '_', '^', backwards, m_str_info)
            else:
                free_up, free_down, backwards = XypicDiagramDrawer._check_free_space_diagonal(dom_i, cod_i, dom_j, cod_j, grid)
                set_label_position(free_up, free_down, '^', '_', backwards, m_str_info)

    @staticmethod
    def _morphism_sort_key(morphism, object_coords):
        """
        Provides a morphism sorting key such that horizontal or
        vertical morphisms between neighbouring objects come
        first, then horizontal or vertical morphisms between more
        far away objects, and finally, all other morphisms.
        """
        i, j = object_coords[morphism.domain]
        target_i, target_j = object_coords[morphism.codomain]
        if morphism.domain == morphism.codomain:
            return (3, 0, default_sort_key(morphism))
        if target_i == i:
            return (1, abs(target_j - j), default_sort_key(morphism))
        if target_j == j:
            return (1, abs(target_i - i), default_sort_key(morphism))
        return (2, 0, default_sort_key(morphism))

    @staticmethod
    def _build_xypic_string(diagram, grid, morphisms, morphisms_str_info, diagram_format):
        """
        Given a collection of :class:`ArrowStringDescription`
        describing the morphisms of a diagram and the object layout
        information of a diagram, produces the final Xy-pic picture.
        """
        object_morphisms = {}
        for obj in diagram.objects:
            object_morphisms[obj] = []
        for morphism in morphisms:
            object_morphisms[morphism.domain].append(morphism)
        result = '\\xymatrix%s{\n' % diagram_format
        for i in range(grid.height):
            for j in range(grid.width):
                obj = grid[i, j]
                if obj:
                    result += latex(obj) + ' '
                    morphisms_to_draw = object_morphisms[obj]
                    for morphism in morphisms_to_draw:
                        result += str(morphisms_str_info[morphism]) + ' '
                if j < grid.width - 1:
                    result += '& '
            if i < grid.height - 1:
                result += '\\\\'
            result += '\n'
        result += '}\n'
        return result

    def draw(self, diagram, grid, masked=None, diagram_format=''):
        """
        Returns the Xy-pic representation of ``diagram`` laid out in
        ``grid``.

        Consider the following simple triangle diagram.

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> from sympy.categories import DiagramGrid, XypicDiagramDrawer
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g], {g * f: "unique"})

        To draw this diagram, its objects need to be laid out with a
        :class:`DiagramGrid`::

        >>> grid = DiagramGrid(diagram)

        Finally, the drawing:

        >>> drawer = XypicDiagramDrawer()
        >>> print(drawer.draw(diagram, grid))
        \\xymatrix{
        A \\ar[d]_{g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\
        C &
        }

        The argument ``masked`` can be used to skip morphisms in the
        presentation of the diagram:

        >>> print(drawer.draw(diagram, grid, masked=[g * f]))
        \\xymatrix{
        A \\ar[r]^{f} & B \\ar[ld]^{g} \\\\
        C &
        }

        Finally, the ``diagram_format`` argument can be used to
        specify the format string of the diagram.  For example, to
        increase the spacing by 1 cm, proceeding as follows:

        >>> print(drawer.draw(diagram, grid, diagram_format="@+1cm"))
        \\xymatrix@+1cm{
        A \\ar[d]_{g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\
        C &
        }

        """
        if not masked:
            morphisms_props = grid.morphisms
        else:
            morphisms_props = {}
            for m, props in grid.morphisms.items():
                if m in masked:
                    continue
                morphisms_props[m] = props
        object_coords = {}
        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    object_coords[grid[i, j]] = (i, j)
        morphisms = sorted(morphisms_props, key=lambda m: XypicDiagramDrawer._morphism_sort_key(m, object_coords))
        morphisms_str_info = {}
        for morphism in morphisms:
            string_description = self._process_morphism(diagram, grid, morphism, object_coords, morphisms, morphisms_str_info)
            if self.default_arrow_formatter:
                self.default_arrow_formatter(string_description)
            for prop in morphisms_props[morphism]:
                if prop.name in self.arrow_formatters:
                    formatter = self.arrow_formatters[prop.name]
                    formatter(string_description)
            morphisms_str_info[morphism] = string_description
        self._push_labels_out(morphisms_str_info, grid, object_coords)
        return XypicDiagramDrawer._build_xypic_string(diagram, grid, morphisms, morphisms_str_info, diagram_format)