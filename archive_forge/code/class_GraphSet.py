from reportlab.lib import colors
from ._Graph import GraphData
class GraphSet:
    """Graph Set.

    Attributes:
     - id        Unique identifier for the set
     - name      String describing the set

    """

    def __init__(self, name=None):
        """Initialize.

        Arguments:
         - name      String identifying the graph set sensibly

        """
        self.id = id
        self._next_id = 0
        self._graphs = {}
        self.name = name

    def new_graph(self, data, name=None, style='bar', color=colors.lightgreen, altcolor=colors.darkseagreen, linewidth=1, center=None, colour=None, altcolour=None, centre=None):
        """Add a GraphData object to the diagram.

        Arguments:
         - data      List of (position, value) int tuples
         - name      String, description of the graph
         - style     String ('bar', 'heat', 'line') describing how the graph
           will be drawn
         - color    colors.Color describing the color to draw all or 'high'
           (some styles) data (overridden by backwards compatible
           argument with UK spelling, colour).
         - altcolor  colors.Color describing the color to draw 'low' (some
           styles) data (overridden by backwards compatible argument
           with UK spelling, colour).
         - linewidth     Float describing linewidth for graph
         - center        Float setting the value at which the x-axis
           crosses the y-axis (overridden by backwards
           compatible argument with UK spelling, centre)

        Add a GraphData object to the diagram (will be stored internally).
        """
        if colour is not None:
            color = colour
        if altcolour is not None:
            altcolor = altcolour
        if centre is not None:
            center = centre
        id = self._next_id
        graph = GraphData(id, data, name, style, color, altcolor, center)
        graph.linewidth = linewidth
        self._graphs[id] = graph
        self._next_id += 1
        return graph

    def del_graph(self, graph_id):
        """Remove a graph from the set, indicated by its id."""
        del self._graphs[graph_id]

    def get_graphs(self):
        """Return list of all graphs in the graph set, sorted by id.

        Sorting is to ensure reliable stacking.
        """
        return [self._graphs[id] for id in sorted(self._graphs)]

    def get_ids(self):
        """Return a list of all ids for the graph set."""
        return list(self._graphs.keys())

    def range(self):
        """Return the lowest and highest base (or mark) numbers as a tuple."""
        lows, highs = ([], [])
        for graph in self._graphs.values():
            low, high = graph.range()
            lows.append(low)
            highs.append(high)
        return (min(lows), max(highs))

    def data_quartiles(self):
        """Return (minimum, lowerQ, medianQ, upperQ, maximum) values as a tuple."""
        data = []
        for graph in self._graphs.values():
            data += list(graph.data.values())
        data.sort()
        datalen = len(data)
        return (data[0], data[datalen / 4], data[datalen / 2], data[3 * datalen / 4], data[-1])

    def to_string(self, verbose=0):
        """Return a formatted string with information about the set.

        Arguments:
            - verbose - Flag indicating whether a short or complete account
              of the set is required

        """
        if not verbose:
            return f'{self}'
        else:
            outstr = [f'\n<{self.__class__}: {self.name}>']
            outstr.append('%d graphs' % len(self._graphs))
            for key in self._graphs:
                outstr.append(f'{self._graphs[key]}')
            return '\n'.join(outstr)

    def __len__(self):
        """Return the number of graphs in the set."""
        return len(self._graphs)

    def __getitem__(self, key):
        """Return a graph, keyed by id."""
        return self._graphs[key]

    def __str__(self):
        """Return a formatted string with information about the feature set."""
        outstr = [f'\n<{self.__class__}: {self.name}>']
        outstr.append('%d graphs' % len(self._graphs))
        outstr = '\n'.join(outstr)
        return outstr