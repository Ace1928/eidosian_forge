from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.mixins.sql import SQLCreatable

    Parent class for creatable, cluster-wide entities (like a database or role). Defines some of
    the abstract methods needed since they are consistent across entities.
    