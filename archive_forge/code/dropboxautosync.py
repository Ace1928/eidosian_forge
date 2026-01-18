import dropbox
import os

# Dropbox access token
ACCESS_TOKEN = "your_access_token"

# Local directory to synchronize
local_directory = "/path/to/local/directory"

# Dropbox directory to synchronize
dropbox_directory = "/path/to/dropbox/directory"

# Create a Dropbox client
dbx = dropbox.Dropbox(ACCESS_TOKEN)

# Synchronize files from local to Dropbox
for root, dirs, files in os.walk(local_directory):
    for filename in files:
        local_path = os.path.join(root, filename)
        relative_path = os.path.relpath(local_path, local_directory)
        dropbox_path = os.path.join(dropbox_directory, relative_path)

        with open(local_path, "rb") as f:
            dbx.files_upload(
                f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite
            )
            print(f"Uploaded: {local_path} -> {dropbox_path}")

# Synchronize files from Dropbox to local
for entry in dbx.files_list_folder(dropbox_directory).entries:
    if isinstance(entry, dropbox.files.FileMetadata):
        dropbox_path = entry.path_display
        local_path = os.path.join(
            local_directory, os.path.relpath(dropbox_path, dropbox_directory)
        )

        if not os.path.exists(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path))

        _, response = dbx.files_download(dropbox_path)
        with open(local_path, "wb") as f:
            f.write(response.content)
            print(f"Downloaded: {dropbox_path} -> {local_path}")

print("File synchronization completed.")
