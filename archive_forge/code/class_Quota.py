from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Quota(_messages.Message):
    """Quota configuration helps to achieve fairness and budgeting in service
  usage.  - Fairness is achieved through the use of short-term quota limits
  that are usually defined over a time window of several seconds or   minutes.
  When such a limit is applied, for example at the user   level, it ensures
  that no single user will monopolize the service   or a given customer's
  allocated portion of it. - Budgeting is achieved through the use of long-
  term quota limits   that are usually defined over a time window of one or
  more   days. These limits help client application developers predict the
  usage and help budgeting.  Quota enforcement uses a simple token-based
  algorithm for resource sharing.  The quota configuration structure is as
  follows:  - `QuotaLimit` defines a single enforceable limit with a specified
  token amount that can be consumed over a specific duration and   applies to
  a particular entity, like a project or an end user. If   the limit applies
  to a user, each user making the request will   get the specified number of
  tokens to consume. When the tokens   run out, the requests from that user
  will be blocked until the   duration elapses and the next duration window
  starts.  - `QuotaGroup` groups a set of quota limits.  - `QuotaRule` maps a
  method to a set of quota groups. This allows   sharing of quota groups
  across methods as well as one method   consuming tokens from more than one
  quota group. When a group   contains multiple limits, requests to a method
  consuming tokens   from that group must satisfy all the limits in that
  group.  Example:      quota:       groups:       - name: ReadGroup
  limits:         - description: Daily Limit           name: ProjectQpd
  default_limit: 10000           duration: 1d           limit_by:
  CLIENT_PROJECT          - description: Per-second Limit           name:
  UserQps           default_limit: 20000           duration: 100s
  limit_by: USER        - name: WriteGroup         limits:         -
  description: Daily Limit           name: ProjectQpd           default_limit:
  1000           max_limit: 1000           duration: 1d           limit_by:
  CLIENT_PROJECT          - description: Per-second Limit           name:
  UserQps           default_limit: 2000           max_limit: 4000
  duration: 100s           limit_by: USER        rules:       - selector: "*"
  groups:         - group: ReadGroup       - selector:
  google.calendar.Calendar.Update         groups:         - group: WriteGroup
  cost: 2       - selector: google.calendar.Calendar.Delete         groups:
  - group: WriteGroup  Here, the configuration defines two quota groups:
  ReadGroup and WriteGroup, each defining its own daily and per-second limits.
  Note that One Platform enforces per-second limits averaged over a duration
  of 100 seconds. The rules map ReadGroup for all methods, except for the
  Update and Delete methods. These two methods consume from WriteGroup, with
  Update method consuming at twice the rate as Delete method.  Multiple quota
  groups can be specified for a method. The quota limits in all of those
  groups will be enforced. Example:      quota:       groups:       - name:
  WriteGroup         limits:         - description: Daily Limit
  name: ProjectQpd           default_limit: 1000           max_limit: 1000
  duration: 1d           limit_by: CLIENT_PROJECT          - description: Per-
  second Limit           name: UserQps           default_limit: 2000
  max_limit: 4000           duration: 100s           limit_by: USER        -
  name: StorageGroup         limits:         - description: Storage Quota
  name: StorageQuota           default_limit: 1000           duration: 0
  limit_by: USER        rules:       - selector:
  google.calendar.Calendar.Create         groups:         - group:
  StorageGroup         - group: WriteGroup       - selector:
  google.calendar.Calendar.Delete         groups:         - group:
  StorageGroup  In the above example, the Create and Delete methods manage the
  user's storage space. In addition, Create method uses WriteGroup to manage
  the requests. In this case, requests to Create method need to satisfy all
  quota limits defined in both quota groups.  One can disable quota for
  selected method(s) identified by the selector by setting disable_quota to
  ture. For example,        rules:       - selector: "*"         group:
  - group ReadGroup       - selector: google.calendar.Calendar.Select
  disable_quota: true

  Fields:
    groups: List of `QuotaGroup` definitions for the service.
    rules: List of `QuotaRule` definitions, each one mapping a selected method
      to one or more quota groups.
  """
    groups = _messages.MessageField('QuotaGroup', 1, repeated=True)
    rules = _messages.MessageField('QuotaRule', 2, repeated=True)