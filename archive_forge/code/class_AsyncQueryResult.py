import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
class AsyncQueryResult(QueryResult):
    """
    Async version for the QueryResult class - a class that
    represents a result of the query operation.
    """

    def __init__(self):
        """
        To init the class you must call self.initialize()
        """
        pass

    async def initialize(self, graph, response, profile=False):
        """
        Initializes the class.
        Args:

        graph:
            The graph on which the query was executed.
        response:
            The response from the server.
        profile:
            A boolean indicating if the query command was "GRAPH.PROFILE"
        """
        self.graph = graph
        self.header = []
        self.result_set = []
        self._check_for_errors(response)
        if len(response) == 1:
            self.parse_statistics(response[0])
        elif profile:
            self.parse_profile(response)
        else:
            self.parse_statistics(response[-1])
            await self.parse_results(response)
        return self

    async def parse_node(self, cell):
        """
        Parses a node from the cell.
        """
        labels = None
        if len(cell[1]) > 0:
            labels = []
            for inner_label in cell[1]:
                labels.append(await self.graph.get_label(inner_label))
        properties = await self.parse_entity_properties(cell[2])
        node_id = int(cell[0])
        return Node(node_id=node_id, label=labels, properties=properties)

    async def parse_scalar(self, cell):
        """
        Parses a scalar value from the server response.
        """
        scalar_type = int(cell[0])
        value = cell[1]
        try:
            scalar = await self.parse_scalar_types[scalar_type](value)
        except TypeError:
            scalar = self.parse_scalar_types[scalar_type](value)
        return scalar

    async def parse_records(self, raw_result_set):
        """
        Parses the result set and returns a list of records.
        """
        records = []
        for row in raw_result_set[1]:
            record = [await self.parse_record_types[self.header[idx][0]](cell) for idx, cell in enumerate(row)]
            records.append(record)
        return records

    async def parse_results(self, raw_result_set):
        """
        Parse the query execution result returned from the server.
        """
        self.header = self.parse_header(raw_result_set)
        if len(self.header) == 0:
            return
        self.result_set = await self.parse_records(raw_result_set)

    async def parse_entity_properties(self, props):
        """
        Parse node / edge properties.
        """
        properties = {}
        for prop in props:
            prop_name = await self.graph.get_property(prop[0])
            prop_value = await self.parse_scalar(prop[1:])
            properties[prop_name] = prop_value
        return properties

    async def parse_edge(self, cell):
        """
        Parse the cell to an edge.
        """
        edge_id = int(cell[0])
        relation = await self.graph.get_relation(cell[1])
        src_node_id = int(cell[2])
        dest_node_id = int(cell[3])
        properties = await self.parse_entity_properties(cell[4])
        return Edge(src_node_id, relation, dest_node_id, edge_id=edge_id, properties=properties)

    async def parse_path(self, cell):
        """
        Parse the cell to a path.
        """
        nodes = await self.parse_scalar(cell[0])
        edges = await self.parse_scalar(cell[1])
        return Path(nodes, edges)

    async def parse_map(self, cell):
        """
        Parse the cell to a map.
        """
        m = OrderedDict()
        n_entries = len(cell)
        for i in range(0, n_entries, 2):
            key = self.parse_string(cell[i])
            m[key] = await self.parse_scalar(cell[i + 1])
        return m

    async def parse_array(self, value):
        """
        Parse array value.
        """
        scalar = [await self.parse_scalar(value[i]) for i in range(len(value))]
        return scalar