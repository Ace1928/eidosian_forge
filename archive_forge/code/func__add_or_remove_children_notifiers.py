def _add_or_remove_children_notifiers(self):
    """ Recursively add or remove notifiers for the children ObserverGraph.
        """
    for child_graph in self.graph.children:
        for next_object in self.graph.node.iter_objects(self.object):
            add_or_remove_notifiers(object=next_object, graph=child_graph, handler=self.handler, target=self.target, dispatcher=self.dispatcher, remove=self.remove)