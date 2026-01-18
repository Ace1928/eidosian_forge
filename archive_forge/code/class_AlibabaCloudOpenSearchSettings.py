import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
class AlibabaCloudOpenSearchSettings:
    """Alibaba Cloud Opensearch` client configuration.

    Attribute:
        endpoint (str) : The endpoint of opensearch instance, You can find it
         from the console of Alibaba Cloud OpenSearch.
        instance_id (str) : The identify of opensearch instance, You can find
         it from the console of Alibaba Cloud OpenSearch.
        username (str) : The username specified when purchasing the instance.
        password (str) : The password specified when purchasing the instance，
          After the instance is created, you can modify it on the console.
        tablename (str): The table name specified during instance configuration.
        field_name_mapping (Dict) : Using field name mapping between opensearch
          vector store and opensearch instance configuration table field names:
        {
            'id': 'The id field name map of index document.',
            'document': 'The text field name map of index document.',
            'embedding': 'In the embedding field of the opensearch instance,
              the values must be in float type and separated by separator,
              default is comma.',
            'metadata_field_x': 'Metadata field mapping includes the mapped
             field name and operator in the mapping value, separated by a comma
             between the mapped field name and the operator.',
        }
        protocol (str): Communication Protocol between SDK and Server, default is http.
        namespace (str) : The instance data will be partitioned based on the "namespace"
         field,If the namespace is enabled, you need to specify the namespace field
         name during initialization, Otherwise, the queries cannot be executed
         correctly.
        embedding_field_separator(str): Delimiter specified for writing vector
         field data, default is comma.
        output_fields: Specify the field list returned when invoking OpenSearch,
         by default it is the value list of the field mapping field.
    """

    def __init__(self, endpoint: str, instance_id: str, username: str, password: str, table_name: str, field_name_mapping: Dict[str, str], protocol: str='http', namespace: str='', embedding_field_separator: str=',', output_fields: Optional[List[str]]=None) -> None:
        self.endpoint = endpoint
        self.instance_id = instance_id
        self.protocol = protocol
        self.username = username
        self.password = password
        self.namespace = namespace
        self.table_name = table_name
        self.opt_table_name = '_'.join([self.instance_id, self.table_name])
        self.field_name_mapping = field_name_mapping
        self.embedding_field_separator = embedding_field_separator
        if output_fields is None:
            self.output_fields = [field.split(',')[0] for field in self.field_name_mapping.values()]
        self.inverse_field_name_mapping: Dict[str, str] = {}
        for key, value in self.field_name_mapping.items():
            self.inverse_field_name_mapping[value.split(',')[0]] = key

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)