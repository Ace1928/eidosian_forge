import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
class BaseNeptuneGraph(ABC):

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Neptune database"""
        return self.schema

    @abstractmethod
    def query(self, query: str, params: dict={}) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def _get_summary(self) -> Dict:
        raise NotImplementedError()

    def _get_labels(self) -> Tuple[List[str], List[str]]:
        """Get node and edge labels from the Neptune statistics summary"""
        summary = self._get_summary()
        n_labels = summary['nodeLabels']
        e_labels = summary['edgeLabels']
        return (n_labels, e_labels)

    def _get_triples(self, e_labels: List[str]) -> List[str]:
        triple_query = '\n        MATCH (a)-[e:`{e_label}`]->(b)\n        WITH a,e,b LIMIT 3000\n        RETURN DISTINCT labels(a) AS from, type(e) AS edge, labels(b) AS to\n        LIMIT 10\n        '
        triple_template = '(:`{a}`)-[:`{e}`]->(:`{b}`)'
        triple_schema = []
        for label in e_labels:
            q = triple_query.format(e_label=label)
            data = self.query(q)
            for d in data:
                triple = triple_template.format(a=d['from'][0], e=d['edge'], b=d['to'][0])
                triple_schema.append(triple)
        return triple_schema

    def _get_node_properties(self, n_labels: List[str], types: Dict) -> List:
        node_properties_query = '\n        MATCH (a:`{n_label}`)\n        RETURN properties(a) AS props\n        LIMIT 100\n        '
        node_properties = []
        for label in n_labels:
            q = node_properties_query.format(n_label=label)
            data = {'label': label, 'properties': self.query(q)}
            s = set({})
            for p in data['properties']:
                for k, v in p['props'].items():
                    s.add((k, types[type(v).__name__]))
            np = {'properties': [{'property': k, 'type': v} for k, v in s], 'labels': label}
            node_properties.append(np)
        return node_properties

    def _get_edge_properties(self, e_labels: List[str], types: Dict[str, Any]) -> List:
        edge_properties_query = '\n        MATCH ()-[e:`{e_label}`]->()\n        RETURN properties(e) AS props\n        LIMIT 100\n        '
        edge_properties = []
        for label in e_labels:
            q = edge_properties_query.format(e_label=label)
            data = {'label': label, 'properties': self.query(q)}
            s = set({})
            for p in data['properties']:
                for k, v in p['props'].items():
                    s.add((k, types[type(v).__name__]))
            ep = {'type': label, 'properties': [{'property': k, 'type': v} for k, v in s]}
            edge_properties.append(ep)
        return edge_properties

    def _refresh_schema(self) -> None:
        """
        Refreshes the Neptune graph schema information.
        """
        types = {'str': 'STRING', 'float': 'DOUBLE', 'int': 'INTEGER', 'list': 'LIST', 'dict': 'MAP', 'bool': 'BOOLEAN'}
        n_labels, e_labels = self._get_labels()
        triple_schema = self._get_triples(e_labels)
        node_properties = self._get_node_properties(n_labels, types)
        edge_properties = self._get_edge_properties(e_labels, types)
        self.schema = f'\n        Node properties are the following:\n        {node_properties}\n        Relationship properties are the following:\n        {edge_properties}\n        The relationships are the following:\n        {triple_schema}\n        '