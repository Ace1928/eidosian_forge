from boto.resultset import ResultSet
class ResourceRecordSets(ResultSet):
    """
    A list of resource records.

    :ivar hosted_zone_id: The ID of the hosted zone.
    :ivar comment: A comment that will be stored with the change.
    :ivar changes: A list of changes.
    """
    ChangeResourceRecordSetsBody = '<?xml version="1.0" encoding="UTF-8"?>\n    <ChangeResourceRecordSetsRequest xmlns="https://route53.amazonaws.com/doc/2013-04-01/">\n            <ChangeBatch>\n                <Comment>%(comment)s</Comment>\n                <Changes>%(changes)s</Changes>\n            </ChangeBatch>\n        </ChangeResourceRecordSetsRequest>'
    ChangeXML = '<Change>\n        <Action>%(action)s</Action>\n        %(record)s\n    </Change>'

    def __init__(self, connection=None, hosted_zone_id=None, comment=None):
        self.connection = connection
        self.hosted_zone_id = hosted_zone_id
        self.comment = comment
        self.changes = []
        self.next_record_name = None
        self.next_record_type = None
        self.next_record_identifier = None
        super(ResourceRecordSets, self).__init__([('ResourceRecordSet', Record)])

    def __repr__(self):
        if self.changes:
            record_list = ','.join([c.__repr__() for c in self.changes])
        else:
            record_list = ','.join([record.__repr__() for record in self])
        return '<ResourceRecordSets:%s [%s]' % (self.hosted_zone_id, record_list)

    def add_change(self, action, name, type, ttl=600, alias_hosted_zone_id=None, alias_dns_name=None, identifier=None, weight=None, region=None, alias_evaluate_target_health=None, health_check=None, failover=None):
        """
        Add a change request to the set.

        :type action: str
        :param action: The action to perform ('CREATE'|'DELETE'|'UPSERT')

        :type name: str
        :param name: The name of the domain you want to perform the action on.

        :type type: str
        :param type: The DNS record type.  Valid values are:

            * A
            * AAAA
            * CNAME
            * MX
            * NS
            * PTR
            * SOA
            * SPF
            * SRV
            * TXT

        :type ttl: int
        :param ttl: The resource record cache time to live (TTL), in seconds.

        :type alias_hosted_zone_id: str
        :param alias_dns_name: *Alias resource record sets only* The value
            of the hosted zone ID, CanonicalHostedZoneNameId, for
            the LoadBalancer.

        :type alias_dns_name: str
        :param alias_hosted_zone_id: *Alias resource record sets only*
            Information about the domain to which you are redirecting traffic.

        :type identifier: str
        :param identifier: *Weighted and latency-based resource record sets
            only* An identifier that differentiates among multiple resource
            record sets that have the same combination of DNS name and type.

        :type weight: int
        :param weight: *Weighted resource record sets only* Among resource
            record sets that have the same combination of DNS name and type,
            a value that determines what portion of traffic for the current
            resource record set is routed to the associated location

        :type region: str
        :param region: *Latency-based resource record sets only* Among resource
            record sets that have the same combination of DNS name and type,
            a value that determines which region this should be associated with
            for the latency-based routing

        :type alias_evaluate_target_health: bool
        :param alias_evaluate_target_health: *Required for alias resource record
            sets* Indicates whether this Resource Record Set should respect the
            health status of any health checks associated with the ALIAS target
            record which it is linked to.

        :type health_check: str
        :param health_check: Health check to associate with this record

        :type failover: str
        :param failover: *Failover resource record sets only* Whether this is the
            primary or secondary resource record set.
        """
        change = Record(name, type, ttl, alias_hosted_zone_id=alias_hosted_zone_id, alias_dns_name=alias_dns_name, identifier=identifier, weight=weight, region=region, alias_evaluate_target_health=alias_evaluate_target_health, health_check=health_check, failover=failover)
        self.changes.append([action, change])
        return change

    def add_change_record(self, action, change):
        """Add an existing record to a change set with the specified action"""
        self.changes.append([action, change])
        return

    def to_xml(self):
        """Convert this ResourceRecordSet into XML
        to be saved via the ChangeResourceRecordSetsRequest"""
        changesXML = ''
        for change in self.changes:
            changeParams = {'action': change[0], 'record': change[1].to_xml()}
            changesXML += self.ChangeXML % changeParams
        params = {'comment': self.comment, 'changes': changesXML}
        return self.ChangeResourceRecordSetsBody % params

    def commit(self):
        """Commit this change"""
        if not self.connection:
            import boto
            self.connection = boto.connect_route53()
        return self.connection.change_rrsets(self.hosted_zone_id, self.to_xml())

    def endElement(self, name, value, connection):
        """Overwritten to also add the NextRecordName,
        NextRecordType and NextRecordIdentifier to the base object"""
        if name == 'NextRecordName':
            self.next_record_name = value
        elif name == 'NextRecordType':
            self.next_record_type = value
        elif name == 'NextRecordIdentifier':
            self.next_record_identifier = value
        else:
            return super(ResourceRecordSets, self).endElement(name, value, connection)

    def __iter__(self):
        """Override the next function to support paging"""
        results = super(ResourceRecordSets, self).__iter__()
        truncated = self.is_truncated
        while results:
            for obj in results:
                yield obj
            if self.is_truncated:
                self.is_truncated = False
                results = self.connection.get_all_rrsets(self.hosted_zone_id, name=self.next_record_name, type=self.next_record_type, identifier=self.next_record_identifier)
            else:
                results = None
                self.is_truncated = truncated