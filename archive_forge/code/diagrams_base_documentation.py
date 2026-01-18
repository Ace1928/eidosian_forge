import copy
import abc
import logging
import six
 Returns a graph object.
        Args:
            title (str): Title of the generated graph
            roi_state (State): If not None, the returned graph will only contain edges and states connected to it.
        Returns:
             A graph instance with a `draw` that allows to render the graph.
        