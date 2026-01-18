from collections import defaultdict
from copy import deepcopy
import datetime
import io
import itertools
import logging
import os
import shutil
import tempfile
from urllib.parse import urlparse
import dateutil.parser
from prov import Error, serializers
from prov.constants import *
from prov.identifier import Identifier, QualifiedName, Namespace
class ProvBundle(object):
    """PROV Bundle"""

    def __init__(self, records=None, identifier=None, namespaces=None, document=None):
        """
        Constructor.

        :param records: Optional iterable of records to add to the bundle
            (default: None).
        :param identifier: Optional identifier of the bundle (default: None).
        :param namespaces: Optional iterable of :py:class:`~prov.identifier.Namespace`s
            to set the document up with (default: None).
        :param document: Optional document to add to the bundle (default: None).
        """
        self._identifier = identifier
        self._records = list()
        self._id_map = defaultdict(list)
        self._document = document
        self._namespaces = NamespaceManager(namespaces, parent=document._namespaces if document is not None else None)
        if records:
            for record in records:
                self.add_record(record)

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self._identifier)

    @property
    def namespaces(self):
        """
        Returns the set of registered namespaces.

        :return: Set of :py:class:`~prov.identifier.Namespace`.
        """
        return set(self._namespaces.get_registered_namespaces())

    @property
    def default_ns_uri(self):
        """
        Returns the default namespace's URI, if any.

        :return: URI as string.
        """
        default_ns = self._namespaces.get_default_namespace()
        return default_ns.uri if default_ns else None

    @property
    def document(self):
        """
        Returns the parent document, if any.

        :return: :py:class:`ProvDocument`.
        """
        return self._document

    @property
    def identifier(self):
        """
        Returns the bundle's identifier
        """
        return self._identifier

    @property
    def records(self):
        """
        Returns the list of all records in the current bundle
        """
        return list(self._records)

    def set_default_namespace(self, uri):
        """
        Sets the default namespace through a given URI.

        :param uri: Namespace URI.
        """
        self._namespaces.set_default_namespace(uri)

    def get_default_namespace(self):
        """
        Returns the default namespace.

        :return: :py:class:`~prov.identifier.Namespace`
        """
        return self._namespaces.get_default_namespace()

    def add_namespace(self, namespace_or_prefix, uri=None):
        """
        Adds a namespace (if not available, yet).

        :param namespace_or_prefix: :py:class:`~prov.identifier.Namespace` or its
            prefix as a string to add.
        :param uri: Namespace URI (default: None). Must be present if only a
            prefix is given in the previous parameter.
        """
        if uri is None:
            return self._namespaces.add_namespace(namespace_or_prefix)
        else:
            return self._namespaces.add_namespace(Namespace(namespace_or_prefix, uri))

    def get_registered_namespaces(self):
        """
        Returns all registered namespaces.

        :return: Iterable of :py:class:`~prov.identifier.Namespace`.
        """
        return self._namespaces.get_registered_namespaces()

    def valid_qualified_name(self, identifier):
        return self._namespaces.valid_qualified_name(identifier)

    def get_records(self, class_or_type_or_tuple=None):
        """
        Returns all records. Returned records may be filtered by the optional
        argument.

        :param class_or_type_or_tuple: A filter on the type for which records are
            to be returned (default: None). The filter checks by the type of the
            record using the `isinstance` check on the record.
        :return: List of :py:class:`ProvRecord` objects.
        """
        results = list(self._records)
        if class_or_type_or_tuple:
            return filter(lambda rec: isinstance(rec, class_or_type_or_tuple), results)
        else:
            return results

    def get_record(self, identifier):
        """
        Returns a specific record matching a given identifier.

        :param identifier: Record identifier.
        :return: :py:class:`ProvRecord`
        """
        if identifier is None:
            return None
        valid_id = self.valid_qualified_name(identifier)
        try:
            return self._id_map[valid_id]
        except KeyError:
            if self.is_bundle():
                return self.document.get_record(valid_id)
            else:
                return None

    def is_document(self):
        """
        `True` if the object is a document, `False` otherwise.

        :return: bool
        """
        return False

    def is_bundle(self):
        """
        `True` if the object is a bundle, `False` otherwise.

        :return: bool
        """
        return True

    def has_bundles(self):
        """
        `True` if the object has at least one bundle, `False` otherwise.

        :return: bool
        """
        return False

    @property
    def bundles(self):
        """
        Returns bundles contained in the document

        :return: Iterable of :py:class:`ProvBundle`.
        """
        return frozenset()

    def get_provn(self, _indent_level=0):
        """
        Returns the PROV-N representation of the bundle.

        :return: String
        """
        indentation = '' + '  ' * _indent_level
        newline = '\n' + '  ' * (_indent_level + 1)
        lines = ['document'] if self.is_document() else ['bundle %s' % self._identifier]
        default_namespace = self._namespaces.get_default_namespace()
        if default_namespace:
            lines.append('default <%s>' % default_namespace.uri)
        registered_namespaces = self._namespaces.get_registered_namespaces()
        if registered_namespaces:
            lines.extend(['prefix %s <%s>' % (namespace.prefix, namespace.uri) for namespace in registered_namespaces])
        if default_namespace or registered_namespaces:
            lines.append('')
        lines.extend([record.get_provn() for record in self._records])
        if self.is_document():
            lines.extend((bundle.get_provn(_indent_level + 1) for bundle in self.bundles))
        provn_str = newline.join(lines) + '\n'
        provn_str += indentation + ('endDocument' if self.is_document() else 'endBundle')
        return provn_str

    def __eq__(self, other):
        if not isinstance(other, ProvBundle):
            return False
        other_records = set(other.get_records())
        this_records = set(self.get_records())
        if len(this_records) != len(other_records):
            return False
        for record_a in this_records:
            found = False
            for record_b in other_records:
                if record_a == record_b:
                    other_records.remove(record_b)
                    found = True
                    break
            if not found:
                logger.debug('Equality (ProvBundle): Could not find this record: %s', str(record_a))
                return False
        return True

    def __ne__(self, other):
        return not self == other
    __hash__ = None

    def _unified_records(self):
        """Returns a list of unified records."""
        merged_records = dict()
        for identifier, records in self._id_map.items():
            if len(records) > 1:
                merged = records[0].copy()
                for record in records[1:]:
                    merged.add_attributes(record.attributes)
                for record in records:
                    merged_records[record] = merged
        if not merged_records:
            return list(self._records)
        added_merged_records = set()
        unified_records = list()
        for record in self._records:
            if record in merged_records:
                merged = merged_records[record]
                if merged not in added_merged_records:
                    unified_records.append(merged)
                    added_merged_records.add(merged)
            else:
                unified_records.append(record)
        return unified_records

    def unified(self):
        """
        Unifies all records in the bundle that haves same identifiers

        :returns: :py:class:`ProvBundle` -- the new unified bundle.
        """
        unified_records = self._unified_records()
        bundle = ProvBundle(records=unified_records, identifier=self.identifier)
        return bundle

    def update(self, other):
        """
        Append all the records of the *other* ProvBundle into this bundle.

        :param other: the other bundle whose records to be appended.
        :type other: :py:class:`ProvBundle`
        :returns: None.
        """
        if isinstance(other, ProvBundle):
            if other.is_document() and other.has_bundles():
                raise ProvException('ProvBundle.update(): The other bundle is a document with sub-bundle(s).')
            for record in other.get_records():
                self.add_record(record)
        else:
            raise ProvException('ProvBundle.update(): The other bundle is not a ProvBundle instance (%s)' % type(other))

    def _add_record(self, record):
        identifier = record.identifier
        if identifier is not None:
            self._id_map[identifier].append(record)
        self._records.append(record)

    def new_record(self, record_type, identifier, attributes=None, other_attributes=None):
        """
        Creates a new record.

        :param record_type: Type of record (one of :py:const:`PROV_REC_CLS`).
        :param identifier: Identifier for new record.
        :param attributes: Attributes as a dictionary or list of tuples to be added
            to the record optionally (default: None).
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        attr_list = []
        if attributes:
            if isinstance(attributes, dict):
                attr_list.extend(((attr, value) for attr, value in attributes.items()))
            else:
                attr_list.extend(attributes)
        if other_attributes:
            attr_list.extend(other_attributes.items() if isinstance(other_attributes, dict) else other_attributes)
        new_record = PROV_REC_CLS[record_type](self, self.valid_qualified_name(identifier), attr_list)
        self._add_record(new_record)
        return new_record

    def add_record(self, record):
        """
        Adds a new record that to the bundle.

        :param record: :py:class:`ProvRecord` to be added.
        """
        return self.new_record(record.get_type(), record.identifier, record.formal_attributes, record.extra_attributes)

    def entity(self, identifier, other_attributes=None):
        """
        Creates a new entity.

        :param identifier: Identifier for new entity.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_ENTITY, identifier, None, other_attributes)

    def activity(self, identifier, startTime=None, endTime=None, other_attributes=None):
        """
        Creates a new activity.

        :param identifier: Identifier for new activity.
        :param startTime: Optional start time for the activity (default: None).
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param endTime: Optional start time for the activity (default: None).
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_ACTIVITY, identifier, {PROV_ATTR_STARTTIME: _ensure_datetime(startTime), PROV_ATTR_ENDTIME: _ensure_datetime(endTime)}, other_attributes)

    def generation(self, entity, activity=None, time=None, identifier=None, other_attributes=None):
        """
        Creates a new generation record for an entity.

        :param entity: Entity or a string identifier for the entity.
        :param activity: Activity or string identifier of the activity involved in
            the generation (default: None).
        :param time: Optional time for the generation (default: None).
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param identifier: Identifier for new generation record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_GENERATION, identifier, {PROV_ATTR_ENTITY: entity, PROV_ATTR_ACTIVITY: activity, PROV_ATTR_TIME: _ensure_datetime(time)}, other_attributes)

    def usage(self, activity, entity=None, time=None, identifier=None, other_attributes=None):
        """
        Creates a new usage record for an activity.

        :param activity: Activity or a string identifier for the entity.
        :param entity: Entity or string identifier of the entity involved in
            the usage relationship (default: None).
        :param time: Optional time for the usage (default: None).
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param identifier: Identifier for new usage record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_USAGE, identifier, {PROV_ATTR_ACTIVITY: activity, PROV_ATTR_ENTITY: entity, PROV_ATTR_TIME: _ensure_datetime(time)}, other_attributes)

    def start(self, activity, trigger=None, starter=None, time=None, identifier=None, other_attributes=None):
        """
        Creates a new start record for an activity.

        :param activity: Activity or a string identifier for the entity.
        :param trigger: Entity triggering the start of this activity.
        :param starter: Optionally extra activity to state a qualified start
            through which the trigger entity for the start is generated
            (default: None).
        :param time: Optional time for the start (default: None).
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param identifier: Identifier for new start record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_START, identifier, {PROV_ATTR_ACTIVITY: activity, PROV_ATTR_TRIGGER: trigger, PROV_ATTR_STARTER: starter, PROV_ATTR_TIME: _ensure_datetime(time)}, other_attributes)

    def end(self, activity, trigger=None, ender=None, time=None, identifier=None, other_attributes=None):
        """
        Creates a new end record for an activity.

        :param activity: Activity or a string identifier for the entity.
        :param trigger: trigger: Entity triggering the end of this activity.
        :param ender: Optionally extra activity to state a qualified end
            through which the trigger entity for the end is generated
            (default: None).
        :param time: Optional time for the end (default: None).
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param identifier: Identifier for new end record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_END, identifier, {PROV_ATTR_ACTIVITY: activity, PROV_ATTR_TRIGGER: trigger, PROV_ATTR_ENDER: ender, PROV_ATTR_TIME: _ensure_datetime(time)}, other_attributes)

    def invalidation(self, entity, activity=None, time=None, identifier=None, other_attributes=None):
        """
        Creates a new invalidation record for an entity.

        :param entity: Entity or a string identifier for the entity.
        :param activity: Activity or string identifier of the activity involved in
            the invalidation (default: None).
        :param time: Optional time for the invalidation (default: None).
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param identifier: Identifier for new invalidation record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_INVALIDATION, identifier, {PROV_ATTR_ENTITY: entity, PROV_ATTR_ACTIVITY: activity, PROV_ATTR_TIME: _ensure_datetime(time)}, other_attributes)

    def communication(self, informed, informant, identifier=None, other_attributes=None):
        """
        Creates a new communication record for an entity.

        :param informed: The informed activity (relationship destination).
        :param informant: The informing activity (relationship source).
        :param identifier: Identifier for new communication record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_COMMUNICATION, identifier, {PROV_ATTR_INFORMED: informed, PROV_ATTR_INFORMANT: informant}, other_attributes)

    def agent(self, identifier, other_attributes=None):
        """
        Creates a new agent.

        :param identifier: Identifier for new agent.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_AGENT, identifier, None, other_attributes)

    def attribution(self, entity, agent, identifier=None, other_attributes=None):
        """
        Creates a new attribution record between an entity and an agent.

        :param entity: Entity or a string identifier for the entity (relationship
            source).
        :param agent: Agent or string identifier of the agent involved in the
            attribution (relationship destination).
        :param identifier: Identifier for new attribution record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_ATTRIBUTION, identifier, {PROV_ATTR_ENTITY: entity, PROV_ATTR_AGENT: agent}, other_attributes)

    def association(self, activity, agent=None, plan=None, identifier=None, other_attributes=None):
        """
        Creates a new association record for an activity.

        :param activity: Activity or a string identifier for the activity.
        :param agent: Agent or string identifier of the agent involved in the
            association (default: None).
        :param plan: Optionally extra entity to state qualified association through
            an internal plan (default: None).
        :param identifier: Identifier for new association record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_ASSOCIATION, identifier, {PROV_ATTR_ACTIVITY: activity, PROV_ATTR_AGENT: agent, PROV_ATTR_PLAN: plan}, other_attributes)

    def delegation(self, delegate, responsible, activity=None, identifier=None, other_attributes=None):
        """
        Creates a new delegation record on behalf of an agent.

        :param delegate: Agent delegating the responsibility (relationship source).
        :param responsible: Agent the responsibility is delegated to (relationship
            destination).
        :param activity: Optionally extra activity to state qualified delegation
            internally (default: None).
        :param identifier: Identifier for new association record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_DELEGATION, identifier, {PROV_ATTR_DELEGATE: delegate, PROV_ATTR_RESPONSIBLE: responsible, PROV_ATTR_ACTIVITY: activity}, other_attributes)

    def influence(self, influencee, influencer, identifier=None, other_attributes=None):
        """
        Creates a new influence record between two entities, activities or agents.

        :param influencee: Influenced entity, activity or agent (relationship
            source).
        :param influencer: Influencing entity, activity or agent (relationship
            destination).
        :param identifier: Identifier for new influence record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        return self.new_record(PROV_INFLUENCE, identifier, {PROV_ATTR_INFLUENCEE: influencee, PROV_ATTR_INFLUENCER: influencer}, other_attributes)

    def derivation(self, generatedEntity, usedEntity, activity=None, generation=None, usage=None, identifier=None, other_attributes=None):
        """
        Creates a new derivation record for a generated entity from a used entity.

        :param generatedEntity: Entity or a string identifier for the generated
            entity (relationship source).
        :param usedEntity: Entity or a string identifier for the used entity
            (relationship destination).
        :param activity: Activity or string identifier of the activity involved in
            the derivation (default: None).
        :param generation: Optionally extra activity to state qualified generation
            through a generation (default: None).
        :param usage: XXX (default: None).
        :param identifier: Identifier for new derivation record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        attributes = {PROV_ATTR_GENERATED_ENTITY: generatedEntity, PROV_ATTR_USED_ENTITY: usedEntity, PROV_ATTR_ACTIVITY: activity, PROV_ATTR_GENERATION: generation, PROV_ATTR_USAGE: usage}
        return self.new_record(PROV_DERIVATION, identifier, attributes, other_attributes)

    def revision(self, generatedEntity, usedEntity, activity=None, generation=None, usage=None, identifier=None, other_attributes=None):
        """
        Creates a new revision record for a generated entity from a used entity.

        :param generatedEntity: Entity or a string identifier for the generated
            entity (relationship source).
        :param usedEntity: Entity or a string identifier for the used entity
            (relationship destination).
        :param activity: Activity or string identifier of the activity involved in
            the revision (default: None).
        :param generation: Optionally to state qualified revision through a
            generation activity (default: None).
        :param usage: XXX (default: None).
        :param identifier: Identifier for new revision record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        record = self.derivation(generatedEntity, usedEntity, activity, generation, usage, identifier, other_attributes)
        record.add_asserted_type(PROV['Revision'])
        return record

    def quotation(self, generatedEntity, usedEntity, activity=None, generation=None, usage=None, identifier=None, other_attributes=None):
        """
        Creates a new quotation record for a generated entity from a used entity.

        :param generatedEntity: Entity or a string identifier for the generated
            entity (relationship source).
        :param usedEntity: Entity or a string identifier for the used entity
            (relationship destination).
        :param activity: Activity or string identifier of the activity involved in
            the quotation (default: None).
        :param generation: Optionally to state qualified quotation through a
            generation activity (default: None).
        :param usage: XXX (default: None).
        :param identifier: Identifier for new quotation record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        record = self.derivation(generatedEntity, usedEntity, activity, generation, usage, identifier, other_attributes)
        record.add_asserted_type(PROV['Quotation'])
        return record

    def primary_source(self, generatedEntity, usedEntity, activity=None, generation=None, usage=None, identifier=None, other_attributes=None):
        """
        Creates a new primary source record for a generated entity from a used
        entity.

        :param generatedEntity: Entity or a string identifier for the generated
            entity (relationship source).
        :param usedEntity: Entity or a string identifier for the used entity
            (relationship destination).
        :param activity: Activity or string identifier of the activity involved in
            the primary source (default: None).
        :param generation: Optionally to state qualified primary source through a
            generation activity (default: None).
        :param usage: XXX (default: None).
        :param identifier: Identifier for new primary source record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        record = self.derivation(generatedEntity, usedEntity, activity, generation, usage, identifier, other_attributes)
        record.add_asserted_type(PROV['PrimarySource'])
        return record

    def specialization(self, specificEntity, generalEntity):
        """
        Creates a new specialisation record for a specific from a general entity.

        :param specificEntity: Entity or a string identifier for the specific
            entity (relationship source).
        :param generalEntity: Entity or a string identifier for the general entity
            (relationship destination).
        """
        return self.new_record(PROV_SPECIALIZATION, None, {PROV_ATTR_SPECIFIC_ENTITY: specificEntity, PROV_ATTR_GENERAL_ENTITY: generalEntity})

    def alternate(self, alternate1, alternate2):
        """
        Creates a new alternate record between two entities.

        :param alternate1: Entity or a string identifier for the first entity
            (relationship source).
        :param alternate2: Entity or a string identifier for the second entity
            (relationship destination).
        """
        return self.new_record(PROV_ALTERNATE, None, {PROV_ATTR_ALTERNATE1: alternate1, PROV_ATTR_ALTERNATE2: alternate2})

    def mention(self, specificEntity, generalEntity, bundle):
        """
        Creates a new mention record for a specific from a general entity.

        :param specificEntity: Entity or a string identifier for the specific
            entity (relationship source).
        :param generalEntity: Entity or a string identifier for the general entity
            (relationship destination).
        :param bundle: XXX
        """
        return self.new_record(PROV_MENTION, None, {PROV_ATTR_SPECIFIC_ENTITY: specificEntity, PROV_ATTR_GENERAL_ENTITY: generalEntity, PROV_ATTR_BUNDLE: bundle})

    def collection(self, identifier, other_attributes=None):
        """
        Creates a new collection record for a particular record.

        :param identifier: Identifier for new collection record.
        :param other_attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        record = self.new_record(PROV_ENTITY, identifier, None, other_attributes)
        record.add_asserted_type(PROV['Collection'])
        return record

    def membership(self, collection, entity):
        """
        Creates a new membership record for an entity to a collection.

        :param collection: Collection the entity is to be added to.
        :param entity: Entity to be added to the collection.
        """
        return self.new_record(PROV_MEMBERSHIP, None, {PROV_ATTR_COLLECTION: collection, PROV_ATTR_ENTITY: entity})

    def plot(self, filename=None, show_nary=True, use_labels=False, show_element_attributes=True, show_relation_attributes=True):
        """
        Convenience function to plot a PROV document.

        :param filename: The filename to save to. If not given, it will open
            an interactive matplotlib plot. The filetype is determined from
            the filename ending.
        :type filename: String
        :param show_nary: Shows all elements in n-ary relations.
        :type show_nary: bool
        :param use_labels: Uses the `prov:label` property of an element as its
            name (instead of its identifier).
        :type use_labels: bool
        :param show_element_attributes: Shows attributes of elements.
        :type show_element_attributes: bool
        :param show_relation_attributes: Shows attributes of relations.
        :type show_relation_attributes: bool
        """
        from prov import dot
        if filename:
            format = os.path.splitext(filename)[-1].lower().strip(os.path.extsep)
        else:
            format = 'png'
        format = format.lower()
        d = dot.prov_to_dot(self, show_nary=show_nary, use_labels=use_labels, show_element_attributes=show_element_attributes, show_relation_attributes=show_relation_attributes)
        method = 'create_%s' % format
        if not hasattr(d, method):
            raise ValueError("Format '%s' cannot be saved." % format)
        with io.BytesIO() as buf:
            buf.write(getattr(d, method)())
            buf.seek(0, 0)
            if filename:
                with open(filename, 'wb') as fh:
                    fh.write(buf.read())
            else:
                import matplotlib.pylab as plt
                import matplotlib.image as mpimg
                max_size = 30
                img = mpimg.imread(buf)
                img = img[1:-1, 1:-1]
                size = (img.shape[1] / 100.0, img.shape[0] / 100.0)
                if max(size) > max_size:
                    scale = max_size / max(size)
                else:
                    scale = 1.0
                size = (scale * size[0], scale * size[1])
                plt.figure(figsize=size)
                plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(img)
                plt.axis('off')
                plt.show()
    wasGeneratedBy = generation
    used = usage
    wasStartedBy = start
    wasEndedBy = end
    wasInvalidatedBy = invalidation
    wasInformedBy = communication
    wasAttributedTo = attribution
    wasAssociatedWith = association
    actedOnBehalfOf = delegation
    wasInfluencedBy = influence
    wasDerivedFrom = derivation
    wasRevisionOf = revision
    wasQuotedFrom = quotation
    hadPrimarySource = primary_source
    alternateOf = alternate
    specializationOf = specialization
    mentionOf = mention
    hadMember = membership