from googlecloudsdk.api_lib.audit_manager import util
Generate an Audit Report.

    Args:
      scope: str, the scope to be enrolled.
      eligible_gcs_buckets: str, List of destination among which customer can
        choose to upload their reports during the audit process.
      is_parent_folder: bool, whether the parent is folder and not project.

    Returns:
      Described audit operation resource.
    