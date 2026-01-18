from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogicalTrafficMatrix(_messages.Message):
    """Describes the anticipated network traffic pattern between nodes
  (henceforth referred to as "coordinates") participating in a distributed
  computation. We note the following conventions for coordinates: - Slice
  coordinates are 0-indexed and contiguous with respect to their owning
  traffic matrix. - Chip coordinates are 0-indexed and contiguous relative to
  their owning-slice. Chip coordinates map 1:1 to the physical TPU chips
  deployed in their owning slice.

  Fields:
    customTrafficMatrix: Custom generator syntax used to represent chip-chip
      granularity traffic matrix (TM) that is too large to be reasonably
      represented as a simple adjacency list.
    defaultUniformTrafficMatrix: Generates a default uniform traffic matrix,
      functionally equivalent to a slice-to-slice, all-to-all traffic matrix
      where each slice sends the same amount of traffic to all other slices.
      Notes: - Borg uses this default-generated traffic matrix to achieve
      best-effort, superblock-compact placement for the affinity group. - Borg
      may or may not report default-generated demands to the Datacenter
      Network for proactive provisioning; to depend on this behavior use an
      alternate representation such as `slice_to_slice_adjacency_list`. - This
      representation is likely a good fit for "small" workloads (e.g. 4 VLP
      slices) that wish to benefit from best-effort superblock colocation
      without needing to provide a detailed traffic matrix. - This option is
      likely not a good fit for "large" workloads with non-uniform traffic
      patterns (e.g. a 100 VLP slice workload with 10 data parallel replicas
      communicating in an all-to-all pattern and 10 pipeline stages
      communicating in a bidirection ring pattern), as it does not give Borg
      enough detail on which of the slices should be colocated in the same
      superblock as other slices. At the largest (e.g. cell-wide) scale, this
      devolves into "random" placement.
    sliceToSliceAdjacencyList: Adjacency list representation of a slice->slice
      granularity TM. Slice-to-slice encoding assumes traffic flows out of
      each slice uniformly. This assumption is inconsequential for superblock
      (SB) Network Aware Scheduling (NAS) of slices that fit within a SB (e.g.
      VLP).
  """
    customTrafficMatrix = _messages.MessageField('CustomTrafficMatrix', 1)
    defaultUniformTrafficMatrix = _messages.MessageField('DefaultUniformTrafficMatrix', 2)
    sliceToSliceAdjacencyList = _messages.MessageField('SliceToSliceAdjacencyList', 3)