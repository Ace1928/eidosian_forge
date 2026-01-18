import subprocess

# Local directory to synchronize
local_dir = "/path/to/local/directory/"

# Remote directory to synchronize with
remote_dir = "user@remote_host:/path/to/remote/directory/"

# rsync command with options
rsync_command = ["rsync", "-avz", "--delete", local_dir, remote_dir]

# Execute the rsync command
subprocess.run(rsync_command, check=True)

print("File synchronization completed.")
